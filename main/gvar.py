from __future__ import print_function
import numpy as np
import logging
import os
import sys
import optim
import optim.adamw
import optim.kfac
import optim.ekfac

import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.multiprocessing

import utils
import models
from data import get_loaders, get_minvar_loader
from args import get_opt
from log_utils import TBXWrapper
from log_utils import Profiler
from log_utils import LogCollector
from estim.gvar import MinVarianceGradient
tb_logger = TBXWrapper()
# torch.multiprocessing.set_sharing_strategy('file_system')


class OptimizerFactory(object):

    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
        self.niters = 0
        self.optimizer = None
        self.logger = LogCollector(opt)
        self.param_groups = None
        self.gest_used = False
        self.reset()

    def reset(self):
        model = self.model
        opt = self.opt
        if opt.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=opt.lr, momentum=opt.momentum,
                                        weight_decay=opt.weight_decay,
                                        nesterov=opt.nesterov)
        elif opt.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=opt.lr,
                                         betas=opt.adam_betas,
                                         eps=opt.adam_eps,
                                         weight_decay=opt.weight_decay)
        elif opt.optim == 'adamw':
            optimizer = optim.adamw.AdamW(
                model.parameters(),
                lr=opt.lr,
                betas=opt.adam_betas,
                eps=opt.adam_eps,
                weight_decay=opt.weight_decay,
                l2_reg=False)
        elif opt.optim == 'kfac':
            optimizer = optim.kfac.KFACOptimizer(
                model,
                lr=opt.lr,
                momentum=opt.momentum,
                stat_decay=opt.kf_stat_decay,
                damping=opt.kf_damping,
                kl_clip=opt.kf_kl_clip,
                weight_decay=opt.weight_decay,
                TCov=opt.kf_TCov,
                TInv=opt.kf_TInv)
        elif opt.optim == 'ekfac':
            optimizer = optim.ekfac.EKFACOptimizer(
                model,
                lr=opt.lr,
                momentum=opt.momentum,
                stat_decay=opt.kf_stat_decay,
                damping=opt.kf_damping,
                kl_clip=opt.kf_kl_clip,
                weight_decay=opt.weight_decay,
                TCov=opt.kf_TCov,
                TScal=opt.kf_TScal,
                TInv=opt.kf_TInv)
        self.optimizer = optimizer
        if self.param_groups is not None:
            self.optimizer.param_groups = self.param_groups
        else:
            self.param_groups = self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


def test(tb_logger, model, test_loader,
         opt, niters, set_name='Test', prefix='V'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, idx in test_loader:
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='none')
            test_loss += loss.sum().item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        wrong = len(test_loader.dataset) - correct
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        error = 100. * wrong / len(test_loader.dataset)
        logging.info(
            '\n{0} set: Average loss: {1:.4f}'
            ', Accuracy: {2}/{3} ({4:.2f}%)'
            ', Error: {5}/{3} ({6:.2f}%)\n'.format(
                set_name, test_loss, correct, len(test_loader.dataset),
                accuracy, wrong, error))
        tb_logger.log_value('%sloss' % prefix, test_loss, step=niters)
        tb_logger.log_value('%scorrect' % prefix, correct, step=niters)
        tb_logger.log_value('%swrong' % prefix, wrong, step=niters)
        tb_logger.log_value('%sacc' % prefix, accuracy, step=niters)
        tb_logger.log_value('%serror' % prefix, error, step=niters)
    return accuracy


def train(tb_logger, epoch, train_loader, model, optimizer, opt, test_loader,
          save_checkpoint, train_test_loader, gvar):
    batch_time = Profiler()
    model.train()
    profiler = Profiler()
    epoch_iters = int(np.ceil(1. * len(train_loader.dataset) / opt.batch_size))
    optimizer.logger.reset()
    # for batch_idx, (data, target, idx) in enumerate(train_loader):
    for batch_idx in range(opt.epoch_iters):
        profiler.start()
        # sgd step
        optimizer.zero_grad()
        # if optimizer.niters == 1050:
        #     print('Sampler repermuted')
        #     gvar.gest.init_data_iter()
        pg_used = gvar.gest_used
        loss = gvar.grad(optimizer.niters)
        if gvar.gest_used != pg_used:
            logging.info('Optimizer reset.')
            optimizer.gest_used = gvar.gest_used
            utils.adjust_lr(optimizer, opt)
            optimizer.reset()
        optimizer.step()
        profiler.toc('optim')
        # snapshot
        # if ((optimizer.niters-opt.gvar_start) % opt.g_msnap_iter == 0
        #         and optimizer.niters >= opt.gvar_start):
        if ((optimizer.niters % opt.g_msnap_iter == 0 and opt.g_avg > 1)
                or (optimizer.niters == 0 and opt.g_avg == 1)):
            # Update model snaps
            # logging.info('Snap Model')
            gvar.snap_model(model)
            profiler.toc('snap_model')
        if ((optimizer.niters - opt.gvar_start) % opt.g_osnap_iter == 0
                and optimizer.niters >= opt.gvar_start):
            # Frequent snaps
            # logging.info('Snap Online')
            gvar.snap_online(model, optimizer.niters)
            profiler.toc('snap_online')
        if ((optimizer.niters - opt.gvar_start) % opt.g_bsnap_iter == 0
                and optimizer.niters >= opt.gvar_start):
            # Rare snaps
            logging.info('Batch Snapshot')
            gvar.snap_batch(model, optimizer.niters)
            profiler.toc('snap_batch')
            profiler.end()
            logging.info('%s' % str(profiler))
        profiler.end()

        batch_time.toc('Time')
        batch_time.end()
        optimizer.niters += 1
        niters = optimizer.niters

        # if True:
        if batch_idx % opt.log_interval == 0:
            gvar_log = ''
            prof_log = ''
            if (batch_idx % opt.gvar_log_iter == 0
                    and optimizer.niters >= opt.gvar_start):
                gvar_log = '\t' + gvar.log_var(model, niters)
            if opt.log_profiler:
                prof_log = '\t' + str(profiler)

            logging.info(
                'Epoch: [{0}][{1}/{2}]({niters})\t'
                'Loss: {loss:.6f}\t'
                '{batch_time}\t'
                '{opt_log}{gvar_log}{prof_log}'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=loss.item(),
                    batch_time=str(batch_time),
                    opt_log=str(optimizer.logger),
                    gvar_log=gvar_log,
                    prof_log=prof_log,
                    niters=niters))
        if batch_idx % opt.tblog_interval == 0:
            tb_logger.log_value('epoch', epoch, step=niters)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.log_value('lr', lr, step=niters)
            tb_logger.log_value('niters', niters, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            # tb_logger.log_value('batch_time', batch_time.val,
            #                     step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)
        # pretrained_accuracy = opt.half_trained and optimizer.niters == 1
        # if optimizer.niters % epoch_iters == 0 or pretrained_accuracy:
        if optimizer.niters % epoch_iters == 0:
            if opt.train_accuracy:
                test(tb_logger,
                     model, train_test_loader, opt, optimizer.niters,
                     'Train', 'T')
            prec1 = test(tb_logger,
                         model, test_loader, opt, optimizer.niters)
            save_checkpoint(model, float(prec1), opt, optimizer, gvar=gvar)
            tb_logger.save_log()


def main():
    opt = get_opt()
    tb_logger.configure(opt.logger_name, flush_secs=5, opt=opt)
    logfname = os.path.join(opt.logger_name, 'log.txt')
    logging.basicConfig(
        filename=logfname,
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(opt.d))

    torch.manual_seed(opt.seed)
    if opt.cuda:
        # TODO: remove deterministic
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
    # helps with wide-resnet by reducing memory and time 2x
    cudnn.benchmark = True

    train_loader, test_loader, train_test_loader = get_loaders(opt)
    minvar_loader = get_minvar_loader(train_loader, opt)

    if opt.epoch_iters == 0:
        opt.epoch_iters = int(
            np.ceil(1. * len(train_loader.dataset) / opt.batch_size))
    opt.maxiter = opt.epoch_iters * opt.epochs
    if opt.g_epoch:
        opt.gvar_start *= opt.epoch_iters
        opt.g_bsnap_iter *= opt.epoch_iters
        opt.g_optim_start = (opt.g_optim_start * opt.epoch_iters) + 1
        opt.g_reinit_iter = opt.g_reinit_iter * opt.epoch_iters
    opt.g_reinit_iter = int(opt.g_reinit_iter)

    model = models.init_model(opt)
    gvar = MinVarianceGradient(model, minvar_loader, opt, tb_logger)

    optimizer = OptimizerFactory(model, opt)
    epoch = 0
    save_checkpoint = utils.SaveCheckpoint()

    # optionally resume from a checkpoint
    model_path = os.path.join(opt.resume, opt.ckpt_name)
    if opt.resume != '':
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            best_prec1 = checkpoint['best_prec1']
            if opt.g_resume:
                gvar.load_state_dict(checkpoint['gvar'])
            else:
                epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model'])
                save_checkpoint.best_prec1 = best_prec1
            print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
                  .format(model_path, epoch, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    if opt.niters > 0:
        max_iters = opt.niters
    else:
        max_iters = opt.epochs * opt.epoch_iters
    while optimizer.niters < max_iters:
        optimizer.epoch = epoch
        utils.adjust_lr(optimizer, opt)
        ecode = train(
            tb_logger,
            epoch, train_loader, model, optimizer, opt, test_loader,
            save_checkpoint, train_test_loader, gvar)
        if ecode == -1:
            break
        epoch += 1
    tb_logger.save_log()


if __name__ == '__main__':
    main()
