from __future__ import print_function
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import logging
from log_utils import LogCollector
from data import get_loaders
import yaml
import os
import glob
import models
# import nonacc_autograd
import utils
from optim.dmom import DMomSGD
from optim.add_dmom import AddDMom
from optim.jvp import DMomSGDJVP, DMomSGDNoScheduler
from log_utils import TBXWrapper
import torch.backends.cudnn as cudnn
from train import train
from test import test, stats
from args import add_args
from gluster.gluster import GradientClusterOnline
tb_logger = TBXWrapper()


def main():
    args = add_args()
    yaml_path = os.path.join('options/{}/{}'.format(args.dataset,
                                                    args.path_opt))
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle)
    od = vars(args)
    for k, v in list(od.items()):
        opt[k] = v
    opt = utils.DictWrapper(opt)

    if opt.lr_scale:
        opt.lr = opt.lr * opt.batch_size/128.

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()
    if opt.optim == 'dmom_ns':
        opt.scheduler = True
    if opt.scheduler:
        opt.sampler = True

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5, opt=opt)

    logging.info(str(opt.d))

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    # helps with wide-resnet by reducing memory and time 2x
    cudnn.benchmark = True

    train_loader, test_loader, train_test_loader = get_loaders(opt)
    if not hasattr(train_loader.dataset.ds, 'classes'):
        train_loader.dataset.ds.classes = range(opt.num_class)
    tb_logger.log_obj('classes', train_loader.dataset.ds.classes)

    epoch_iters = int(np.ceil(1.*len(train_loader.dataset)/opt.batch_size))
    opt.maxiter = epoch_iters * opt.epochs

    model = models.init_model(opt)

    if opt.label_smoothing > 0:
        smooth_label = LabelSmoothing(opt)
        model.criterion = smooth_label.criterion
    else:
        model.criterion = F.nll_loss

    gluster = None
    if opt.gluster:
        gluster = GradientClusterOnline(
                model, opt.gluster_num, opt.gluster_beta)

    if opt.optim == 'dmom':
        optimizer = DMomSGD(model.parameters(),
                            opt,
                            train_size=len(train_loader.dataset),
                            lr=opt.lr, momentum=opt.momentum,
                            dmom=opt.dmom,
                            weight_decay=opt.weight_decay)
    elif opt.optim == 'dmom_jvp':
        optimizer = DMomSGDJVP(model.parameters(),
                               opt,
                               train_size=len(train_loader.dataset),
                               lr=opt.lr, momentum=opt.momentum,
                               weight_decay=opt.weight_decay)
    elif opt.optim == 'dmom_ns':
        optimizer = DMomSGDNoScheduler(model.parameters(),
                                       opt,
                                       train_size=len(train_loader.dataset),
                                       lr=opt.lr, momentum=opt.momentum,
                                       weight_decay=opt.weight_decay)
    elif opt.optim == 'ssgd':
        # optimizer = SimpleSGD(model.parameters(),
        #                       lr=opt.lr, momentum=opt.momentum)
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr, momentum=opt.momentum)
    elif opt.optim == 'adddmom':
        optimizer = AddDMom(model.parameters(),
                            opt,
                            train_size=len(train_loader.dataset),
                            lr=opt.lr, momentum=opt.momentum,
                            dmom=opt.dmom)
    elif opt.optim == 'sgd':
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
    if opt.scheduler:
        optimizer.scheduler = train_loader.sampler.scheduler
        optimizer.scheduler.logger = optimizer.logger
    if opt.sampler:
        train_loader.sampler.logger = optimizer.logger
    optimizer.gluster = gluster
    epoch = 0
    save_checkpoint = utils.SaveCheckpoint()

    # optionally resume from a checkpoint
    if opt.resume:
        resume = glob.glob(opt.resume)
        resume = resume[0] if len(resume) > 0 else opt.resume
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.niters = checkpoint['niters']
            best_prec1 = checkpoint['best_prec1']
            save_checkpoint.best_prec1 = best_prec1
            print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
                  .format(resume, epoch, best_prec1))
            if opt.train_accuracy:
                test(tb_logger, model, gluster,
                     train_test_loader, opt, optimizer.niters, 'Train', 'T')
            test(tb_logger, model, gluster, test_loader, opt, optimizer.niters)
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    count_nz = []
    count_lr_update = 0
    # for epoch in range(opt.epochs):
    while optimizer.niters < opt.epochs*epoch_iters:
        optimizer.epoch = epoch
        if opt.sampler_lr_update:
            count_nz += [int((train_loader.sampler.count == 0).sum())]
            sw = opt.sampler_lr_window
            cavg = sum(count_nz[-sw:])/sw
            lavg = sum(count_nz[-sw/2:])/(sw/2)
            if len(count_nz) > 10 and cavg == lavg:
                if count_lr_update == 0:
                    count_nz = []
                    utils.adjust_learning_rate_count(optimizer)
                    count_lr_update += 1
                else:
                    break
        elif isinstance(opt.lr_decay_epoch, str):
            utils.adjust_learning_rate_multi(
                    optimizer, optimizer.niters//epoch_iters, opt)
        else:
            utils.adjust_learning_rate(
                    optimizer, optimizer.niters//epoch_iters, opt)
        ecode = train(tb_logger,
                      epoch, train_loader, model, optimizer, opt, test_loader,
                      save_checkpoint, train_test_loader)
        if ecode == -1:
            break
        epoch += 1

        # if opt.train_accuracy:
        #     test(model, train_loader, opt, optimizer.niters, 'Train', 'T')
        # test(model, test_loader, opt, optimizer.niters)

    if opt.optim == 'dmom' or opt.optim == 'dmom_jvp':
        if opt.log_image:
            vis_samples(optimizer.alpha, train_loader.dataset, opt.epochs)

        optimizer.logger.reset()
        optimizer.log_perc('last_')
        logging.info(str(optimizer.logger))
        optimizer.logger.tb_log(tb_logger, step=optimizer.niters)

    if opt.scheduler:
        scheduler = optimizer.scheduler
        if opt.log_image:
            vis_samples(scheduler.alpha, train_loader.dataset, opt.epochs)

        scheduler.logger.reset()
        scheduler.log_perc('last_')
        logging.info(str(scheduler.logger))
        scheduler.logger.tb_log(tb_logger, step=optimizer.niters)

    if opt.log_stats:
        stats(tb_logger,
              model, test_loader, opt, optimizer, 'Test', prefix='V')
        stats(tb_logger,
              model, train_test_loader, opt, optimizer, 'Train', prefix='T')
    tb_logger.save_log()


def fix_bn(m):
    if type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm1d:
        m.eval()


class LabelSmoothing:
    # https://github.com/whr94621/NJUNMT-pytorch/blob/aff968c0da9273dc42eabbb8ac4e459f9195f6e4/src/modules/criterions.py#L131
    def __init__(self, opt):
        self.num_class = opt.num_class
        self.label_smoothing = opt.label_smoothing
        self.confidence = 1.0 - opt.label_smoothing

    def criterion(self, outputs, target, reduction='elementwise_mean'):
        one_hot = torch.randn(1, self.num_class)
        one_hot.fill_(self.label_smoothing / (self.num_class - 1))

        tdata = target.detach()
        if outputs.is_cuda:
            one_hot = one_hot.cuda()
        tmp_ = one_hot.repeat(target.size(0), 1)
        tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
        target = tmp_.detach()
        loss = F.kl_div(outputs, target, reduction=reduction)
        return loss


def vis_samples(alphas, train_dataset, epoch):
    big_alphas = np.argsort(alphas)[-9:][::-1]
    small_alphas = np.argsort(alphas)[:9]

    x = [train_dataset[i][0] for i in big_alphas]
    x = vutils.make_grid(x, nrow=3, normalize=True, scale_each=True)
    tb_logger.log_img('big_alpha' % i, x, epoch)
    x = [train_dataset[i][0] for i in small_alphas]
    x = vutils.make_grid(x, nrow=3, normalize=True, scale_each=True)
    tb_logger.log_img('small_alpha' % i, x, epoch)


if __name__ == '__main__':
    main()
