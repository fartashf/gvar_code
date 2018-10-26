from __future__ import print_function
import argparse
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import logging
from log_utils import LogCollector
import shutil
from data import get_loaders
import yaml
import os
import glob
import models
import models.mnist
import models.cifar10
import models.logreg
import models.imagenet
import models.cifar10_wresnet
import models.cifar10_wresnet2
# import nonacc_autograd
from optim.dmom import DMomSGD
from optim.add_dmom import AddDMom
from optim.jvp import DMomSGDJVP, DMomSGDNoScheduler
from log_utils import TBXWrapper
import torch.backends.cudnn as cudnn
from train import train
from test import test, stats
from gluster import GradientCluster
tb_logger = TBXWrapper()


def add_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # options overwritting yaml options
    parser.add_argument('--path_opt', default='default.yaml',
                        type=str, help='path to a yaml options file')
    parser.add_argument('--data', default=argparse.SUPPRESS,
                        type=str, help='path to data')
    parser.add_argument('--logger_name', default='runs/runX')
    parser.add_argument('--dataset', default='mnist', help='mnist|cifar10')

    # options that can be changed from default
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size',
                        type=int, default=argparse.SUPPRESS, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=argparse.SUPPRESS,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=argparse.SUPPRESS,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='how many batches to wait before logging training'
                        ' status')
    parser.add_argument('--tblog_interval',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--optim', default=argparse.SUPPRESS, help='sgd|dmom')
    parser.add_argument('--dmom', type=float, default=argparse.SUPPRESS,
                        help='Data momentum')
    parser.add_argument('--low_theta', type=float, default=argparse.SUPPRESS,
                        help='Low threshold for discarding small norms.')
    parser.add_argument('--high_theta', type=float, default=argparse.SUPPRESS,
                        help='High threshold for discarding large norms.')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default=argparse.SUPPRESS,
                        help='model architecture: (default: resnet32)')
    parser.add_argument('-j', '--workers', default=argparse.SUPPRESS,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--weight_decay', '--wd', default=argparse.SUPPRESS,
                        type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--dmom_interval', default=argparse.SUPPRESS,
                        type=int,
                        metavar='W', help='Update dmom every X epochs.')
    parser.add_argument('--dmom_temp', default=argparse.SUPPRESS,
                        type=float,
                        metavar='W', help='Temperature [0, +inf),'
                        '0 is always dmom, +inf is never dmom')
    parser.add_argument('--alpha', default=argparse.SUPPRESS,
                        help='normg|jacobian')
    parser.add_argument('--jacobian_lambda', default=.5,
                        type=float)
    parser.add_argument('--dmom_theta', type=float, default=0,
                        help='max of dmom/normg')  # x/10, (x/10)^(2)
    parser.add_argument('--sampler', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--train_accuracy', action='store_true',
                        default=argparse.SUPPRESS)
    # parser.add_argument('--save_for_notebooks', action='store_true',
    #                     default=False)
    parser.add_argument('--alpha_norm', default=argparse.SUPPRESS,
                        help='sum_norm|exp_norm|nonorm')
    parser.add_argument('--sampler_weight', default=argparse.SUPPRESS,
                        help='dmom|alpha|normg')
    # parser.add_argument('--wmomentum', action='store_true',
    #                     default=argparse.SUPPRESS)
    parser.add_argument('--log_profiler', action='store_true')
    parser.add_argument('--log_image', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--lr_decay_epoch',
                        default=argparse.SUPPRESS)
    parser.add_argument('--norm_temp', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_alpha_th',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_alpha_perc',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_w2c', default=argparse.SUPPRESS,
                        help='1_over|log')
    parser.add_argument('--sampler_max_count',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--sampler_start_epoch',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_update_interval',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_update_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--sampler_lr_update',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--sampler_lr_window',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--divlen',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--wmomentum', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_keys',
                        default='touch,touch_p,alpha_normed_h,count_h,tau')
    parser.add_argument('--sampler_params', default='10,90,5,100')
    parser.add_argument('--sampler_repetition',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--exp_lr',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--corrupt_perc',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--log_nex',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--lr_scale',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--sampler_maxw',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--data_aug',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--wnoise',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--wnoise_stddev',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--min_batches',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--tauXstd',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--tauXstdL',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--scheduler',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--pretrained',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--nodropout',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--num_class',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--minvar',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--instantw',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--lr_decay_rate',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sma_momentum',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--nesterov',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--log_stats',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--label_smoothing',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--duplicate',
                        default=argparse.SUPPRESS, type=str)
    parser.add_argument('--alpha_diff',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--gluster',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--gluster_num',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gluster_beta',
                        default=argparse.SUPPRESS, type=float)
    args = parser.parse_args()
    return args


def main():
    args = add_args()
    yaml_path = os.path.join('options/{}/{}'.format(args.dataset,
                                                    args.path_opt))
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle)
    od = vars(args)
    for k, v in od.iteritems():
        opt[k] = v
    opt = DictWrapper(opt)

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

    if opt.dataset == 'mnist':
        if opt.arch == 'cnn':
            model = models.mnist.Convnet(not opt.nodropout)
        elif opt.arch == 'bigcnn':
            model = models.mnist.BigConvnet(not opt.nodropout)
        elif opt.arch == 'mlp':
            model = models.mnist.MLP(not opt.nodropout)
        elif opt.arch == 'smlp':
            model = models.mnist.SmallMLP(not opt.nodropout)
        elif opt.arch == 'ssmlp':
            model = models.mnist.SuperSmallMLP(not opt.nodropout)
        # else:
        #     model = models.mnist.MNISTNet()
    elif opt.dataset == 'cifar10' or opt.dataset == 'svhn':
        # model = torch.nn.DataParallel(
        #     models.cifar10.__dict__[opt.arch]())
        # model.cuda()
        if opt.arch == 'cnn':
            model = models.cifar10.Convnet()
        elif opt.arch.startswith('wrn'):
            depth, widen_factor = map(int, opt.arch[3:].split('-'))
            # model = models.cifar10_wresnet.Wide_ResNet(28, 10, 0.3, 10)
            model = models.cifar10_wresnet2.WideResNet(
                depth, opt.num_class, widen_factor, 0.3)
        else:
            model = models.cifar10.__dict__[opt.arch]()
        model = torch.nn.DataParallel(model)
    elif opt.dataset == 'imagenet':
        model = models.imagenet.Model(opt.arch, opt.pretrained)
    elif opt.dataset.startswith('imagenet'):
        model = models.imagenet.Model(opt.arch, opt.pretrained, opt.num_class)
    elif opt.dataset == 'logreg':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '10class':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '5class':
        model = models.logreg.Linear(opt.dim, opt.num_class)

    if opt.cuda:
        model.cuda()

    if opt.label_smoothing > 0:
        smooth_label = LabelSmoothing(opt)
        model.criterion = smooth_label.criterion
    else:
        model.criterion = F.nll_loss

    gluster = None
    if opt.gluster:
        gluster = GradientCluster(model, opt.gluster_num, opt.gluster_beta)

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
    save_checkpoint = SaveCheckpoint()

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
                    adjust_learning_rate_count(optimizer)
                    count_lr_update += 1
                else:
                    break
        elif isinstance(opt.lr_decay_epoch, str):
            adjust_learning_rate_multi(optimizer,
                                       optimizer.niters//epoch_iters, opt)
        else:
            adjust_learning_rate(optimizer, optimizer.niters//epoch_iters, opt)
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


class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]


def vis_samples(alphas, train_dataset, epoch):
    big_alphas = np.argsort(alphas)[-9:][::-1]
    small_alphas = np.argsort(alphas)[:9]

    x = [train_dataset[i][0] for i in big_alphas]
    x = vutils.make_grid(x, nrow=3, normalize=True, scale_each=True)
    tb_logger.log_img('big_alpha' % i, x, epoch)
    x = [train_dataset[i][0] for i in small_alphas]
    x = vutils.make_grid(x, nrow=3, normalize=True, scale_each=True)
    tb_logger.log_img('small_alpha' % i, x, epoch)


def adjust_learning_rate(optimizer, epoch, opt):
    """ Sets the learning rate to the initial LR decayed by 10 """
    if opt.exp_lr:
        """ test
        A=np.arange(200);
        np.round(np.power(.1, np.power(2., A/80.)-1), 6)[[0,80,120,160]]
        test """
        last_epoch = 2. ** (float(epoch) / int(opt.lr_decay_epoch)) - 1
    else:
        last_epoch = epoch // int(opt.lr_decay_epoch)
    lr = opt.lr * (0.1 ** last_epoch)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_multi(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr_decay_epoch = np.array(map(int, opt.lr_decay_epoch.split(',')))
    if len(lr_decay_epoch) == 1:
        return adjust_learning_rate(optimizer, epoch, opt)
    el = (epoch // lr_decay_epoch)
    ei = np.where(el > 0)[0]
    if len(ei) == 0:
        ei = [0]
    print(el)
    print(ei)
    lr = opt.lr * (opt.lr_decay_rate ** (ei[-1] + el[ei[-1]]))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_niters(optimizer, niters, opt, train_size):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (niters // (opt.lr_decay_epoch*train_size)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_count(optimizer):
    """Sets the learning rate to the initial LR decayed by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= .1


class SaveCheckpoint(object):
    def __init__(self):
        # remember best prec@1 and save checkpoint
        self.best_prec1 = 0

    def __call__(self, model, prec1, opt, optimizer,
                 filename='checkpoint.pth.tar'):
        is_best = prec1 > self.best_prec1
        self.best_prec1 = max(prec1, self.best_prec1)
        state = {
            'epoch': optimizer.epoch + 1,
            'niters': optimizer.niters,
            'opt': opt.d,
            'model': model.state_dict(),
            'best_prec1': self.best_prec1,
        }
        if opt.optim == 'dmom' or opt.optim == 'dmom_jvp':
            state.update({
                'weights': optimizer.weights,
                'alpha': optimizer.alpha,
                'alpha_normed': optimizer.alpha_normed,
            })
            if opt.optim == 'dmom':
                state.update({
                    'fnorm': optimizer.fnorm,
                    'normg': optimizer.grad_norm
                })

        torch.save(state, opt.logger_name+'/'+filename)
        if is_best:
            shutil.copyfile(opt.logger_name+'/'+filename,
                            opt.logger_name+'/model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
