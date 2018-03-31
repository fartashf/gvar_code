import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import numpy as np


def get_loaders(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loaders(opt)
    elif opt.dataset == 'cifar10':
        return get_cifar10_loaders(opt)
    elif opt.dataset == 'logreg':
        return get_logreg_loaders(opt)
    elif 'class' in opt.dataset:
        return get_logreg_loaders(opt)


class IndexedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super(IndexedMNIST, self).__getitem__(index)
        return img, target, index


def get_mnist_loaders(opt, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}
    trainset = IndexedMNIST('data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    if opt.sampler:
        weights = torch.ones(len(trainset)).double()
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size,
            sampler=DmomWeightedRandomSampler(weights, len(trainset)),
            **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        IndexedMNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=opt.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


class IndexedCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super(IndexedCIFAR10, self).__getitem__(index)
        return img, target, index


def get_cifar10_loaders(opt):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        IndexedCIFAR10(root='./data', train=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize,
                       ]), download=True),
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        IndexedCIFAR10(root='./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           normalize,
                       ])),
        batch_size=opt.test_batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    return train_loader, test_loader


class DmomWeightedRandomSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples,
                                      self.replacement))

    def __len__(self):
        return self.num_samples


def random_orthogonal_matrix(gain, shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                           "supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.asarray(gain * q, dtype=np.float)


class IndexdLinearData(data.Dataset):
    def __init__(self, C, D, num, dim, num_class, train=True):
        X = np.zeros((C.shape[0], num))
        Y = np.zeros((num,))
        for i in range(num_class):
            n = num/num_class
            e = np.random.normal(0.0, 1.0, (dim, n))
            X[:, i*n:(i+1)*n] = np.dot(D[:, :, i], e) + C[:, i:i+1]
            Y[i*n:(i+1)*n] = i
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        X = torch.Tensor(self.X[:, index]).float()
        Y = int(self.Y[index])
        return X, Y, index

    def __len__(self):
        return self.X.shape[1]


def get_logreg_loaders(opt, **kwargs):
    np.random.seed(1234)
    # print("Create W")
    C = opt.c_const*random_orthogonal_matrix(1.0, (opt.dim, opt.num_class))
    D = opt.d_const*random_orthogonal_matrix(1.0,
                                             (opt.dim, opt.dim, opt.num_class))
    # print("Create train")
    trainset = IndexdLinearData(C, D, opt.num_train_data, opt.dim,
                                opt.num_class, train=True)
    # print("Create test")
    testset = IndexdLinearData(C, D, opt.num_test_data, opt.dim, opt.num_class,
                               train=False)
    torch.save((trainset.X, trainset.Y, testset.X, testset.Y,
                C), opt.logger_name+'/data.pth.tar')

    if opt.sampler:
        weights = torch.ones(len(trainset)).double()
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size,
            sampler=DmomWeightedRandomSampler(weights, len(trainset)),
            **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=opt.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader
