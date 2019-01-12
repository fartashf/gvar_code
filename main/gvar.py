from __future__ import print_function
import numpy as np
import logging
import yaml
import os
import glob
import sys

import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing

import utils
import models
from data import get_loaders
from args import add_args
from log_utils import TBXWrapper
from log_utils import Profiler
from log_utils import LogCollector
from estim.gvar import MinVarianceGradient
tb_logger = TBXWrapper()
torch.multiprocessing.set_sharing_strategy('file_system')


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

        wrong = len(test_loader.dataset)-correct
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
    batch_time = Profiler(k=100)
    model.train()
    profiler = Profiler()
    epoch_iters = int(np.ceil(1.*len(train_loader.dataset)/opt.batch_size))
    optimizer.logger.reset()
    # for batch_idx, (data, target, idx) in enumerate(train_loader):
    for batch_idx in range(opt.epoch_iters):
        profiler.start()
        # sgd step
        optimizer.zero_grad()
        loss = gvar.grad(optimizer.niters)
        optimizer.step()
        optimizer.niters += 1
        profiler.toc('optim')
        # snapshot
        if optimizer.niters % opt.g_osnap_iter == 0:
            # Frequent snaps
            gvar.snap_online(model, optimizer.niters)
            profiler.toc('snap_online')
        if ((optimizer.niters-opt.gvar_start) % opt.g_bsnap_iter == 0
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
        niters = optimizer.niters

        # if True:
        if batch_idx % opt.log_interval == 0:
            gvar_log = ''
            prof_log = ''
            if batch_idx % opt.gvar_log_iter == 0:
                gvar_log = '\t'+gvar.log_var(model, niters)
            if opt.log_profiler:
                prof_log = '\t'+str(profiler)

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
        if optimizer.niters % epoch_iters == 0:
            if opt.train_accuracy:
                test(tb_logger,
                     model, train_test_loader, opt, optimizer.niters,
                     'Train', 'T')
            prec1 = test(tb_logger,
                         model, test_loader, opt, optimizer.niters)
            save_checkpoint(model, float(prec1), opt, optimizer)
            tb_logger.save_log()


def main():
    args = add_args()
    yaml_path = os.path.join('options/{}/{}'.format(args.dataset,
                                                    args.path_opt))
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle)
    od = vars(args)
    for k, v in od.items():
        opt[k] = v
    opt = utils.DictWrapper(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

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

    if opt.epoch_iters == 0:
        opt.epoch_iters = int(
                np.ceil(1.*len(train_loader.dataset)/opt.batch_size))
    opt.maxiter = opt.epoch_iters * opt.epochs

    model = models.init_model(opt)

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr, momentum=opt.momentum,
                              weight_decay=opt.weight_decay,
                              nesterov=opt.nesterov)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=opt.lr,
                               weight_decay=opt.weight_decay)
    optimizer.niters = 0
    optimizer.logger = LogCollector(opt)
    epoch = 0
    save_checkpoint = utils.SaveCheckpoint()

    # optionally resume from a checkpoint
    model_path = os.path.join(opt.run_dir, opt.ckpt_name)
    if opt.resume:
        resume = glob.glob(opt.resume)
        resume = resume[0] if len(resume) > 0 else opt.resume
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            best_prec1 = checkpoint['best_prec1']
            save_checkpoint.best_prec1 = best_prec1
            print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
                  .format(model_path, epoch, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    gvar = MinVarianceGradient(model, train_loader, opt, tb_logger)
    while optimizer.niters < opt.epochs*opt.epoch_iters:
        optimizer.epoch = epoch
        if isinstance(opt.lr_decay_epoch, str):
            utils.adjust_learning_rate_multi(
                    optimizer, optimizer.niters//opt.epoch_iters, opt)
        else:
            utils.adjust_learning_rate(
                    optimizer, optimizer.niters//opt.epoch_iters, opt)
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
