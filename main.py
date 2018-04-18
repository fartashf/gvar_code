from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
import logging
from log_utils import AverageMeter, LogCollector, Profiler
import time
from data import get_loaders
import yaml
import os
import models
import models.mnist
import models.cifar10
import models.logreg
# import nonacc_autograd
from optim import DMomSGD, AddDMom
from log_utils import TBXWrapper
tb_logger = TBXWrapper()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # options overwritting yaml options
    parser.add_argument('--path_opt', default='default.yaml',
                        type=str, help='path to a yaml options file')
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
    parser.add_argument('--lr_decay_epoch', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--norm_temp', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_alpha_th',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_alpha_perc',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_weight_to_count', default=argparse.SUPPRESS,
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
                        default='touch,touch_p,alpha_normed_h,count_h')
    parser.add_argument('--sampler_linear_params', default='10,90,5,100')
    args = parser.parse_args()

    yaml_path = os.path.join('options/{}/{}'.format(args.dataset,
                                                    args.path_opt))
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle)
    od = vars(args)
    for k, v in od.iteritems():
        opt[k] = v
    opt = DictWrapper(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    train_loader, test_loader = get_loaders(opt)

    if opt.dataset == 'mnist':
        if opt.arch == 'convnet':
            model = models.mnist.Convnet()
        elif opt.arch == 'mlp':
            model = models.mnist.MLP()
        else:
            model = models.mnist.MNISTNet()
    elif opt.dataset == 'cifar10':
        model = torch.nn.DataParallel(
            models.cifar10.__dict__[opt.arch]())
        model.cuda()
    elif opt.dataset == 'logreg':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '10class':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '5class':
        model = models.logreg.Linear(opt.dim, opt.num_class)

    if opt.cuda:
        model.cuda()

    if opt.optim == 'dmom':
        optimizer = DMomSGD(model.parameters(),
                            opt,
                            train_size=len(train_loader.dataset),
                            lr=opt.lr, momentum=opt.momentum,
                            dmom=opt.dmom, update_interval=opt.dmom_interval,
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
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr, momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    optimizer.niters = 0

    epoch_iters = len(train_loader.dataset)/opt.batch_size
    count_nz = []
    epoch = 0
    count_lr_update = 0
    # for epoch in range(opt.epochs):
    while optimizer.niters < opt.epochs*epoch_iters:
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
        else:
            adjust_learning_rate(optimizer, optimizer.niters//epoch_iters, opt)
        train(epoch, train_loader, model, optimizer, opt, test_loader)
        epoch += 1

        # if opt.train_accuracy:
        #     test(model, train_loader, opt, optimizer.niters, 'Train', 'T')
        # test(model, test_loader, opt, optimizer.niters)

    if opt.optim == 'dmom':
        if opt.log_image:
            vis_samples(optimizer.alpha, train_loader, opt.epochs)

        optimizer.logger = LogCollector(opt)
        optimizer.log_perc('last_')
        logging.info(str(optimizer.logger))
        optimizer.logger.tb_log(tb_logger, step=opt.epochs)

        torch.save({
            'opt': opt.d,
            'model': model.state_dict(),
            'fnorm': optimizer.fnorm,
            'weights': optimizer.weights,
            'alpha': optimizer.alpha,
            'alpha_normed': optimizer.alpha_normed,
            'normg': optimizer.grad_norm
        }, opt.logger_name+'/model.pth.tar')


def train(epoch, train_loader, model, optimizer, opt, test_loader):
    batch_time = AverageMeter()
    optimizer.logger = LogCollector(opt)
    optimizer.profiler = Profiler()
    model.train()
    end = time.time()
    # update sampler weights
    if opt.sampler and epoch >= opt.sampler_start_epoch:
        if opt.sampler_weight == 'dmom':
            alpha_val = optimizer.data_momentum
        elif opt.sampler_weight == 'alpha':
            alpha_val = optimizer.alpha
        elif opt.sampler_weight == 'normg':
            alpha_val = optimizer.grad_norm
        elif opt.sampler_weight == 'alpha_normed':
            alpha_val = optimizer.alpha_normed
        elif opt.sampler_weight == 'alpha_batch_exp':
            alpha_val = optimizer.alpha
            alpha_val -= alpha_val.max()
            alpha_val = np.exp(alpha_val/opt.norm_temp)
            alpha_val /= alpha_val.sum()
        elif opt.sampler_weight == 'alpha_batch_sum':
            alpha_val = optimizer.alpha
            alpha_val /= alpha_val.sum()
        # optimizer.weights += 1e-5
        # optimizer.weights /= optimizer.weights.sum()
        optimizer.weights = train_loader.sampler.update(alpha_val,
                                                        optimizer.niters)
        count = train_loader.sampler.count
        count_nz = float((count == 0).sum())
        optimizer.logger.update('touch_p', count_nz*100./len(count), 1)
        optimizer.logger.update('touch', count_nz, 1)
        optimizer.logger.update('count_h', count, 1, hist=True)
        optimizer.logger.update('weights_h', optimizer.weights, 1, hist=True)
        optimizer.logger.update('visits_h',
                                train_loader.sampler.visits, 1, hist=True)
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        optimizer.profiler.start()
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if opt.optim == 'dmom':
            optimizer.epoch = epoch
            optimizer.profiler.tic()
            loss = F.nll_loss(output, target, reduce=False)
            optimizer.profiler.toc('forward')
            grads = nonacc_grad(model, loss)
            optimizer.profiler.toc('backward')
            optimizer.step(idx, grads, loss, target)
        elif opt.optim == 'ssgd':
            loss = F.nll_loss(output, target, reduce=False)
            grads = nonacc_grad(model, loss)
            # grads = nonacc_grad_backward(model, loss)
            acc_grad(model, grads)
            optimizer.step()
        elif opt.optim == 'adddmom':
            loss_i = F.nll_loss(output, target, reduce=False)
            loss = optimizer.step(idx, loss_i)
        else:
            optimizer.profiler.tic()
            loss = F.nll_loss(output, target)
            optimizer.profiler.toc('forward')
            # grad = torch.ones_like(loss)
            # nonacc_autograd.backward([loss], [grad])
            # import ipdb; ipdb.set_trace()
            loss.backward()
            optimizer.profiler.toc('backward')
            optimizer.step()
            optimizer.profiler.toc('step')
        optimizer.niters += 1
        optimizer.profiler.end()

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % opt.log_interval == 0:
            niters = optimizer.niters
            loss = loss.mean()
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss:.6f}\t'
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                '{opt_log}'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=loss.data[0],
                    batch_time=batch_time,
                    opt_log=str(optimizer.logger)))
            if opt.log_profiler:
                logging.info(str(optimizer.profiler))
            tb_logger.log_value('epoch', epoch, step=niters)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.log_value('lr', lr, step=niters)
            tb_logger.log_value('niters', niters, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            tb_logger.log_value('batch_time', batch_time.val,
                                step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)
            if np.isnan(float(loss)):
                return

        epoch_iters = len(train_loader.dataset)/opt.batch_size
        if optimizer.niters % epoch_iters == 0:
            if opt.train_accuracy:
                test(model, train_loader, opt, optimizer.niters, 'Train', 'T')
            test(model, test_loader, opt, optimizer.niters)

    if opt.optim == 'dmom':  # and epoch > 9:
        optimizer.logger = LogCollector(opt)
        optimizer.log_perc()
        logging.info(str(optimizer.logger))
        optimizer.logger.tb_log(tb_logger, step=epoch)


def test(model, test_loader, opt, niters, set_name='Test', prefix='V'):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader.sampler.training = False
    for data, target, idx in test_loader:
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    wrong = len(test_loader.dataset)-correct
    test_loss /= len(test_loader.dataset)
    logging.info(
        '\n{0} set: Average loss: {1:.4f}'
        ', Accuracy: {2}/{3} ({4:.2f}%)'
        ', Error: {5}/{3} ({6:.2f}%)\n'.format(
            set_name, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            wrong, 100. * wrong / len(test_loader.dataset)))
    tb_logger.log_value('%sloss' % prefix, test_loss, step=niters)
    tb_logger.log_value('%scorrect' % prefix, correct, step=niters)
    tb_logger.log_value('%swrong' % prefix, wrong, step=niters)
    tb_logger.log_value('%sacc' % prefix,
                        100.*correct/len(test_loader.dataset), step=niters)
    tb_logger.log_value('%serror' % prefix,
                        100.*wrong/len(test_loader.dataset), step=niters)
    test_loader.sampler.training = True


def nonacc_grad(model, loss):
    grads = []
    params = [x for x in model.parameters()]
    for l in loss:
        model.zero_grad()
        tmp_grads = torch.autograd.grad(l, params, retain_graph=True)
        grads.append(tmp_grads)

    # import ipdb; ipdb.set_trace()
    return [grads]


def nonacc_grad_backward(model, loss):
    # https://discuss.pytorch.org/t/quickly-get-individual-gradients-not-sum-of-gradients-of-all-network-outputs/8405
    # https://github.com/tensorflow/tensorflow/issues/4897#issuecomment-253056458
    output_gradients = []

    for output in loss:
        tmp_grad = []
        model.zero_grad()
        output.backward(retain_graph=True)
        for param in model.parameters():
            tmp_grad.append(param.grad.clone())
        output_gradients.append(tmp_grad)

    return [output_gradients]


def acc_grad(model, grads):
    grads = grads[0]
    for i, param in enumerate(model.parameters()):
        param.grad = grads[0][i].clone()
        for j in range(1, len(grads)):
            param.grad += grads[j][i]
        param.grad /= len(grads)


def nonacc_grad_fast(model, loss):
    # http://pytorch.org/docs/master/_modules/torch/autograd.html#grad
    raise NotImplemented()
    params = [x[1] for x in model.named_parameters()]
    grads = []
    for param in params:
        data = param.data
        grads.append(Variable(torch.ones((loss.numel(),)+data.shape),
                              volatile=True).cuda())

    torch.autograd.grad(loss, params, grads)
    return grads


class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]


def vis_samples(alphas, train_loader, epoch):
    big_alphas = np.argsort(alphas)[-9:][::-1]
    small_alphas = np.argsort(alphas)[:9]

    x = [train_loader.dataset[i][0] for i in big_alphas]
    x = vutils.make_grid(x, nrow=3, normalize=True, scale_each=True)
    tb_logger.log_img('big_alpha' % i, x, epoch)
    x = [train_loader.dataset[i][0] for i in small_alphas]
    x = vutils.make_grid(x, nrow=3, normalize=True, scale_each=True)
    tb_logger.log_img('small_alpha' % i, x, epoch)


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.lr_decay_epoch))
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


if __name__ == '__main__':
    main()
