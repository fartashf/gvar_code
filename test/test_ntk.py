import unittest
import numpy as np
import torch
import torch.nn.functional as F
import copy
import sys
import time
import torch.optim as optim
from torchvision import datasets, transforms
import logging
sys.path.append('../')
from models.mnist import MLP, Convnet  # NOQA
from ntk.ntk import NeuralTangentKernel  # NOQA


def set_seed(seed):
    print('seed      : %d' % seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.set_printoptions(suppress=True)
    np.random.seed(seed)


class IndexedDataset():

    def __init__(self, dataset):
        self.ds = dataset

    def __getitem__(self, index):
        img, target = self.ds[index]
        return img, target, index

    def __len__(self):
        return len(self.ds)


def test_mnist(model, batch_size, epochs, **kwargs):
    print('batch_size: %d' % batch_size)
    print('epochs    : %d' % epochs)

    train_dataset = datasets.MNIST(
        './data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    idxdataset = IndexedDataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=batch_size,
        shuffle=True, num_workers=10)
    # shuffle doesn't matter to gluster as long as dataset is returning index

    modelg = copy.deepcopy(model)
    ntk = NeuralTangentKernel(modelg, damping=1e-3, **kwargs)
    # Test if NTK can be disabled
    # gluster.deactivate()

    optimizer = optim.SGD(modelg.parameters(),
                          lr=.01)
    # momentum=0.9,
    # weight_decay=.0005)

    do_ntk = True
    tc = []
    for e in range(epochs):
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            tic = time.time()
            # Note modelg
            modelg.train()
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # gluster.zero_data()
            output = modelg(data)
            loss = F.nll_loss(output, target, reduction='none')
            # store stats
            loss0 = loss.mean()
            if do_ntk:
                loss0.backward(retain_graph=True)
                Ki = ntk.get_kernel_inverse()
                # optim
                optimizer.zero_grad()
                n = data.shape[0]
                loss_ntk = Ki.sum(0).dot(loss)/n/n
                loss_ntk.backward()
            else:
                loss0.backward()
                loss_ntk = torch.tensor(0)
            optimizer.step()
            toc = time.time()
            tc += [toc - tic]
            if batch_idx % 10 == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Loss0: {loss0:.6f}\t Loss NTK: {loss_ntk:.6f}'.format(
                        e, batch_idx, len(train_loader),
                        loss0=loss0.item(), loss_ntk=loss_ntk.item()))
                # print(Ki.shape)
                # print(Ki[:10, :10])
                # torch.cuda.empty_cache()
    print(np.mean(tc))


class MNISTTest(object):
    def test_mnist(self, **kwargs):
        model = self.model
        kwargs.update(self.kwargs)
        epochs = 2
        test_mnist(model, 128, epochs, **kwargs)


class TestNTKMLP(unittest.TestCase, MNISTTest):
    def setUp(self):
        print('In method %s' % self._testMethodName)
        logging.basicConfig(
                format='%(asctime)s %(message)s', level=logging.INFO)
        set_seed(1234)
        self.model = MLP(dropout=False)
        self.model.cuda()
        print('Model initialized.')
        self.kwargs = {}


class TestNTKConv(unittest.TestCase, MNISTTest):
    def setUp(self):
        print('In method %s' % self._testMethodName)
        logging.basicConfig(
                format='%(asctime)s %(message)s', level=logging.INFO)
        set_seed(1234)
        self.model = Convnet(dropout=False)
        self.model.cuda()
        print('Model initialized.')
        self.kwargs = {}


if __name__ == '__main__':
    unittest.main()
