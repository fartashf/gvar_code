from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import logging


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training'
                        ' status')
    parser.add_argument('--optim', default='sgd', help='sgd|dmom')
    parser.add_argument('--dmom', type=float, default=0.5,
                        help='Data momentum')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        NewMNIST('data', train=True, download=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                 ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        NewMNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    if args.cuda:
        model.cuda()

    if args.optim == 'dmom':
        optimizer = DMomSGD(model.parameters(),
                            train_size=len(train_loader.dataset),
                            lr=args.lr, momentum=args.momentum,
                            dmom=args.dmom)
    elif args.optim == 'ssgd':
        # optimizer = SimpleSGD(model.parameters(),
        #                       lr=args.lr, momentum=args.momentum)
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, model, optimizer, args)
        test(model, test_loader, args)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(epoch, train_loader, model, optimizer, args):
    model.train()
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            if args.optim == 'dmom':
                loss = F.nll_loss(output, target, reduce=False)
                grads = nonacc_grad(model, loss)
                optimizer.step(idx, grads)
            elif args.optim == 'ssgd':
                loss = F.nll_loss(output, target, reduce=False)
                grads = nonacc_grad(model, loss)
                # grads = nonacc_grad_backward(model, loss)
                acc_grad(model, grads)
                optimizer.step()
            else:
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
            if batch_idx % args.log_interval == 0:
                loss = loss.mean()
                logging.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target, idx in test_loader:
        if args.cuda:
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


class NewMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super(NewMNIST, self).__getitem__(index)
        return img, target, index


class DMomSGD(optim.Optimizer):
    # http://pytorch.org/docs/master/_modules/torch/optim/sgd.html
    def __init__(self, params, train_size=0, lr=0, momentum=0, dmom=0):
        defaults = dict(lr=lr, momentum=momentum, dmom=dmom)
        super(DMomSGD, self).__init__(params, defaults)
        self.data_momentum = torch.zeros(train_size).cuda()

    def step(self, idx, grads_group, **kwargs):
        # import numpy as np
        # print(np.histogram(self.data_momentum.cpu().numpy()))
        data_mom = self.data_momentum
        for group, grads in zip(self.param_groups, grads_group):
            dmom = group['dmom']
            grad_acc = [torch.zeros_like(g.data).cuda() for g in grads[0]]
            for i in range(len(idx)):
                gf = [g.data.view(-1) for g in grads[i]]
                gc = torch.cat(gf)
                normg = torch.pow(gc, 2).sum(dim=0, keepdim=True).sqrt()
                data_mom[idx[i]:idx[i]+1] = data_mom[idx[i]]*dmom + normg*(
                    1-dmom)
                if float(normg) < 1e-5:
                    print('small norm')
                    continue
                for g, ga in zip(grads[i], grad_acc):
                    ga += g.data*float(data_mom[idx[i]:idx[i]+1]/normg)
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


if __name__ == '__main__':
    main()
