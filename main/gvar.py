from __future__ import print_function
import numpy as np
import logging
import yaml
import os
import time
import glob
import copy

import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing

import utils
import models
from data import get_loaders, InfiniteLoader
from args import add_args
from gluster.gluster import GradientClusterBatch
from log_utils import TBXWrapper
from log_utils import AverageMeter, Profiler
from log_utils import LogCollector
tb_logger = TBXWrapper()
torch.multiprocessing.set_sharing_strategy('file_system')


class GradientEstimator(object):
    def __init__(self, model, train_loader, opt, *args, **kwargs):
        self.opt = opt
        self.model = model
        self.train_loader = train_loader

    def get_grad(self, model_new):
        raise NotImplemented('get_grad not implemented')


class SGDEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SGDEstimator, self).__init__(*args, **kwargs)
        # many open files? torch.multiprocessing sharing file_system
        self.data_iter = iter(InfiniteLoader(self.train_loader))

    def get_grad(self, model):
        data = next(self.data_iter)

        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        g = torch.autograd.grad(loss, model.parameters())
        return g


class GlusterEstimator(SGDEstimator):
    def __init__(self, *args, **kwargs):
        # get_grad is the same as SGD
        super(GlusterEstimator, self).__init__(*args, **kwargs)
        opt = self.opt
        model = self.model
        self.gluster = GradientClusterBatch(
                model, opt.g_min_size, nclusters=opt.g_nclusters,
                no_grad=opt.g_no_grad, active_only=opt.g_active_only)
        self.update()

    def update(self):
        opt = self.opt
        citers = opt.gb_citers
        gluster_tc = np.zeros(citers)
        # TODO: refactor this
        for i in range(citers):
            tic = time.time()
            stat = self.gluster.update_batch(
                    self.train_loader, len(self.train_loader.dataset),
                    ci=i, citers=citers)
            toc = time.time()
            gluster_tc[i] = (toc - tic)
            self.gluster.print_stats()
            logging.info(
                    '%.4f +/- %.4f'
                    % (gluster_tc.mean(), gluster_tc.std()))
        self.train_loader.sampler.set_assign_i(stat[1])


class SVRGEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SVRGEstimator, self).__init__(*args, **kwargs)
        self.model = copy.deepcopy(self.model)
        self.data_iter = iter(InfiniteLoader(self.train_loader))
        model = self.model
        self.mu = [torch.zeros_like(g) for g in model.parameters()]
        num = 0
        for batch_idx, (data, target, idx) in enumerate(self.train_loader):
            num += len(idx)
            model.eval()
            data, target = data.cuda(), target.cuda()
            # model.train() # TODO: SVRG might have trouble with dropout
            model.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            grad_params = torch.autograd.grad(loss, model.parameters())
            for m, g in zip(self.mu, grad_params):
                m += g
        for m in self.mu:
            m /= num

    def get_grad(self, model_new):
        data = next(self.data_iter)

        model_old = self.model
        data, target = data[0].cuda(), data[1].cuda()

        # old grad
        model_old.zero_grad()
        output = model_old(data)
        loss = F.nll_loss(output, target)
        g_old = torch.autograd.grad(loss, model_old.parameters())

        # new grad
        output = model_new(data)
        loss = F.nll_loss(output, target)
        g_new = torch.autograd.grad(loss, model_new.parameters())

        ge = [m-go+gn for m, go, gn in zip(self.mu, g_old, g_new)]
        return ge


def test_gvar(tb_logger, model, train_loader, opt, niters):
    sgd = SGDEstimator(model, train_loader, opt)
    # gluster or SVRG
    if opt.g_estim == 'gluster':
        gest = GlusterEstimator(model, train_loader, opt)
    elif opt.g_estim == 'svrg':
        gest = SVRGEstimator(model, train_loader, opt)
    elif opt.g_estim == 'sgd':
        gest = SGDEstimator(model, train_loader, opt)

    # estimate grad var and bias
    gviter = opt.gvar_iter
    Ege = [torch.zeros_like(g) for g in model.parameters()]
    Esgd = [torch.zeros_like(g) for g in model.parameters()]
    for i in range(gviter):
        ge = gest.get_grad(model)
        for e, g in zip(Ege, ge):
            e += g

        g_sgd = sgd.get_grad(model)
        for e, g in zip(Esgd, g_sgd):
            e += g
    for e, g in zip(Ege, Esgd):
        e /= gviter
        g /= gviter
    bias = np.sum([(ee-gg).pow(2).sum().item() for ee, gg in zip(Ege, Esgd)])
    tb_logger.log_value('grad_bias', float(bias), step=niters)
    nw = sum([w.numel() for w in model.parameters()])
    var_e = 0
    var_s = 0
    for i in range(gviter):
        ge = gest.get_grad(model)
        v = sum([(ee-gg).pow(2).sum() for ee, gg in zip(Ege, ge)])
        var_e += v/nw

        g_sgd = sgd.get_grad(model)
        v = sum([(ee-gg).pow(2).sum() for ee, gg in zip(Esgd, g_sgd)])
        var_s += v/nw
    var_e /= gviter
    var_s /= gviter
    tb_logger.log_value('est_var', float(var_e), step=niters)
    tb_logger.log_value('sgd_var', float(var_s), step=niters)
    logging.info(
            'Gradient bias: %.8f\t sgd var: %.8f\t est var: %.8f'
            % (bias, var_s, var_e))


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
          save_checkpoint, train_test_loader):
    batch_time = AverageMeter()
    optimizer.profiler = Profiler()
    model.train()
    end = time.time()
    epoch_iters = int(np.ceil(1.*len(train_loader.dataset)/opt.batch_size))
    optimizer.logger.reset()
    # for batch_idx, (data, target, idx) in enumerate(train_loader):
    data_iter = iter(InfiniteLoader(train_loader))
    for batch_idx in range(opt.epoch_iters):
        data, target, idx = next(data_iter)
        # train model
        model.train()
        optimizer.profiler.start()
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        optimizer.profiler.tic()
        loss = model.criterion(output, target)
        optimizer.profiler.toc('forward')
        loss.backward()
        optimizer.profiler.toc('backward')
        # optim_log(optimizer, optimizer.logger)
        optimizer.step()
        optimizer.profiler.toc('step')
        optimizer.niters += 1
        optimizer.profiler.end()

        batch_time.update(time.time() - end)
        end = time.time()
        niters = optimizer.niters
        loss = loss.mean()
        if batch_idx % opt.log_interval == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss:.6f}\t'
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                '{opt_log}'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=loss.item(),
                    batch_time=batch_time,
                    opt_log=str(optimizer.logger)))
            if opt.log_profiler:
                logging.info(str(optimizer.profiler))
        if batch_idx % opt.tblog_interval == 0:
            tb_logger.log_value('epoch', epoch, step=niters)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.log_value('lr', lr, step=niters)
            tb_logger.log_value('niters', niters, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            tb_logger.log_value('batch_time', batch_time.val,
                                step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)

        # if True:
        if optimizer.niters % epoch_iters == 0:
            if opt.train_accuracy:
                test(tb_logger,
                     model, train_test_loader, opt, optimizer.niters,
                     'Train', 'T')
            prec1 = test(tb_logger,
                         model, test_loader, opt, optimizer.niters)
            test_gvar(tb_logger, model, train_loader, opt, niters)
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

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5, opt=opt)

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
    model.criterion = F.nll_loss

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr, momentum=opt.momentum,
                              weight_decay=opt.weight_decay,
                              nesterov=opt.nesterov)
    elif opt.optim == 'adam':
        optimizer = optim.SGD(model.parameters(),
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

    while optimizer.niters < opt.epochs*opt.epoch_iters:
        optimizer.epoch = epoch
        if isinstance(opt.lr_decay_epoch, str):
            utils.adjust_learning_rate_multi(
                    optimizer, optimizer.niters//opt.epoch_iters, opt)
        else:
            utils.adjust_learning_rate(
                    optimizer, optimizer.niters//opt.epoch_iters, opt)
        ecode = train(tb_logger,
                      epoch, train_loader, model, optimizer, opt, test_loader,
                      save_checkpoint, train_test_loader)
        if ecode == -1:
            break
        epoch += 1
    tb_logger.save_log()


if __name__ == '__main__':
    main()
