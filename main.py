from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from log_utils import AverageMeter, LogCollector
import tensorboard_logger as tb_logger
import numpy as np
import time
from data import get_loaders
import yaml
import os
import models
import models.mnist
import models.cifar10


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
        model = models.mnist.MNISTNet()
    elif opt.dataset == 'cifar10':
        model = model = torch.nn.DataParallel(
            models.cifar10.__dict__[args.arch]())
        model.cuda()

    if opt.cuda:
        model.cuda()

    if opt.optim == 'dmom':
        optimizer = DMomSGD(model.parameters(),
                            train_size=len(train_loader.dataset),
                            lr=opt.lr, momentum=opt.momentum,
                            dmom=opt.dmom)
    elif opt.optim == 'ssgd':
        # optimizer = SimpleSGD(model.parameters(),
        #                       lr=opt.lr, momentum=opt.momentum)
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr, momentum=opt.momentum)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr, momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    optimizer.niters = 0

    for epoch in range(opt.epochs):
        train(epoch, train_loader, model, optimizer, opt)
        test(model, test_loader, opt, optimizer.niters)


def train(epoch, train_loader, model, optimizer, opt):
    batch_time = AverageMeter()
    optimizer.logger = LogCollector()
    model.train()
    end = time.time()
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if opt.optim == 'dmom':
            optimizer.epoch = epoch
            loss = F.nll_loss(output, target, reduce=False)
            grads = nonacc_grad(model, loss)
            optimizer.step(idx, grads)
        elif opt.optim == 'ssgd':
            loss = F.nll_loss(output, target, reduce=False)
            grads = nonacc_grad(model, loss)
            # grads = nonacc_grad_backward(model, loss)
            acc_grad(model, grads)
            optimizer.step()
        else:
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        optimizer.niters += 1

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
            tb_logger.log_value('epoch', epoch, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            tb_logger.log_value('batch_time', batch_time.val,
                                step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)


def test(model, test_loader, opt, niters):
    model.eval()
    test_loss = 0
    correct = 0
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

    test_loss /= len(test_loader.dataset)
    logging.info(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    tb_logger.log_value('Vloss', test_loss, step=niters)
    tb_logger.log_value('Vcorrect', correct, step=niters)
    tb_logger.log_value('Vwrong', len(test_loader.dataset)-correct,
                        step=niters)
    tb_logger.log_value('Vacc', 100.*correct/len(test_loader.dataset),
                        step=niters)


class DMomSGD(optim.Optimizer):
    # http://pytorch.org/docs/master/_modules/torch/optim/sgd.html
    def __init__(self, params, train_size=0, lr=0, momentum=0, dmom=0,
                 low_theta=1e-5, high_theta=1e5):
        defaults = dict(lr=lr, momentum=momentum, dmom=dmom)
        super(DMomSGD, self).__init__(params, defaults)
        self.data_momentum = torch.ones(train_size).cuda()
        self.grad_norm = torch.ones((train_size,)).cuda()
        self.epoch = 0
        self.low_theta = low_theta
        self.high_theta = high_theta

    def step(self, idx, grads_group, **kwargs):
        # import numpy as np
        # print(np.histogram(self.data_momentum.cpu().numpy()))
        data_mom = self.data_momentum
        numz = 0
        bigg = 0
        normg_ratio = np.zeros((len(idx),))
        dmom_ratio = np.zeros((len(idx),))
        newdmom_normg_ratio = np.zeros((len(idx),))
        normg_val = np.zeros((len(idx),))
        dmom_val = np.zeros((len(idx),))
        for group, grads in zip(self.param_groups, grads_group):
            dmom = group['dmom']
            grad_acc = [torch.zeros_like(g.data).cuda() for g in grads[0]]
            for i in range(len(idx)):
                gf = [g.data.view(-1) for g in grads[i]]
                gc = torch.cat(gf)

                # current normg, dmom
                normg = torch.pow(gc, 2).sum(dim=0, keepdim=True).sqrt()
                new_dmom = data_mom[idx[i]]*dmom + normg*(1-dmom)

                # last normg, dmom
                last_dmom = data_mom[idx[i]:idx[i]+1]
                last_normg = self.grad_norm[idx[i]:idx[i]+1]

                # ratios
                normg_ratio[i] = (normg-last_normg).cpu().numpy().copy()
                dmom_ratio[i] = (new_dmom-last_dmom).cpu().numpy().copy()
                newdmom_normg_ratio[i] = (new_dmom-normg).cpu().numpy().copy()
                normg_val[i] = normg.cpu().numpy().copy()
                dmom_val[i] = new_dmom.cpu().numpy().copy()

                # update normg, dmom
                data_mom[idx[i]:idx[i]+1] = new_dmom
                self.grad_norm[idx[i]:idx[i]+1] = normg

                if float(normg) < self.low_theta:
                    numz += 1
                if float(normg) > self.high_theta:
                    bigg += 1
                    continue
                # accumulate current mini-batch weighted grad
                for g, ga in zip(grads[i], grad_acc):
                    ga += g.data*float(
                        data_mom[idx[i]:idx[i]+1]/(normg+self.low_theta))
                    # ga += g.data

            momentum = group['momentum']
            for p, ga in zip(group['params'], grad_acc):
                param_state = self.state[p]
                d_p = ga/len(idx)
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p)
                d_p = buf
                p.data.add_(-group['lr'], d_p)

        self.logger.update('normg_ratio', normg_ratio[:len(idx)], len(idx))
        self.logger.update('dmom_ratio', dmom_ratio[:len(idx)], len(idx))
        self.logger.update('newdmom_normg_ratio',
                           newdmom_normg_ratio[:len(idx)], len(idx))
        self.logger.update('numz', numz, len(idx))
        self.logger.update('zpercent', numz*100./len(idx), len(idx))
        self.logger.update('bigg', bigg, len(idx))
        self.logger.update('normg', normg_val[:len(idx)], len(idx))
        self.logger.update('dmom', dmom_val[:len(idx)], len(idx))


class SimpleSGD(optim.Optimizer):
    def __init__(self, params, lr=0, momentum=0, **kwargs):
        defaults = dict(lr=lr, momentum=momentum)
        super(SimpleSGD, self).__init__(params, defaults)

    def step(self, **kwargs):
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                param_state = self.state[p]
                d_p = p.grad.data
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p)
                d_p = buf
                p.data.add_(-group['lr'], d_p)


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


if __name__ == '__main__':
    main()
