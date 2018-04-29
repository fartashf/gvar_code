import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import numpy as np
import os


def get_loaders(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loaders(opt)
    elif opt.dataset == 'cifar10':
        return get_cifar10_loaders(opt)
    elif opt.dataset == 'imagenet':
        return get_imagenet_loaders(opt)
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
    trainset = IndexedMNIST(opt.data, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    if opt.sampler:
        # weights = torch.ones(len(trainset)).double()
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size,
            # sampler=DmomWeightedRandomSampler(weights, len(trainset)),
            sampler=DelayedSampler(len(trainset), opt),
            **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        IndexedMNIST(opt.data, train=False, transform=transforms.Compose([
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

    # valid_size=0.1
    # split = int(np.floor(valid_size * num_train))
    # indices = list(range(num_train))
    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataset = IndexedCIFAR10(root=opt.data, train=True,
                                   transform=transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop(32, 4),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]), download=True)
    if opt.sampler:
        train_sampler = DelayedSampler(len(train_dataset), opt)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=opt.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        IndexedCIFAR10(root=opt.data, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           normalize,
                       ])),
        batch_size=opt.test_batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    return train_loader, test_loader


class IndexedImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        img, target = super(IndexedImageFolder, self).__getitem__(index)
        return img, target, index


def get_imagenet_loaders(opt):
    # Data loading code
    traindir = os.path.join(opt.data, 'train')
    valdir = os.path.join(opt.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = IndexedImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if opt.sampler:
        train_sampler = DelayedSampler(len(train_dataset), opt)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        num_workers=opt.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IndexedImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    return train_loader, val_loader


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


class DelayedSampler(Sampler):

    def __init__(self, num_samples, opt):
        self.count = np.zeros(num_samples)
        self.indices = range(num_samples)
        self.num_samples = num_samples
        self.training = True
        self.opt = opt
        self.weights = np.ones(num_samples)
        self.visits = np.zeros(num_samples)
        self.last_count = np.zeros(num_samples)

    def __iter__(self):
        if self.training:
            self.indices = np.where(self.count == 0)[0]
            I = self.indices
            # return (self.indices[i]
            #         for i in torch.randperm(len(self.indices)))
            if self.opt.sampler_repetition:
                W = self.weights[I]
                I = [np.ones(int(w), dtype=int)*i for i, w in zip(I, W)]
                I = np.concatenate(I)
            return (I[i] for i in torch.randperm(len(I)))
        return iter(torch.randperm(self.num_samples))

    def update(self, optimizer):
        opt = self.opt
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
            alpha_val = np.exp(alpha_val / opt.norm_temp)
            alpha_val /= alpha_val.sum()
        elif opt.sampler_weight == 'alpha_batch_sum':
            alpha_val = optimizer.alpha
            alpha_val /= alpha_val.sum()
        alpha = alpha_val

        if self.opt.sampler_weight_to_count == '1_over':
            amx = 1. / self.opt.sampler_max_count
            count = 1. / np.maximum(amx, alpha)
            # count = count-np.min(count)  # min should be 0
            # weights = 1./(count+1)
            # weights = count+1
        elif self.opt.sampler_weight_to_count == 'log':
            amx = np.exp(-self.opt.sampler_max_count)
            count = -np.log(np.maximum(amx, alpha))
            # count = count-np.min(count)  # min should be 0
            # weights = np.exp(-count)
            # weights = count+1
        elif self.opt.sampler_weight_to_count == 'c_times':
            amx = self.opt.sampler_max_count
            alpha = np.minimum(1, np.maximum(0, alpha))  # ensuring in [0,1]
            count = amx * (1 - alpha)
        elif self.opt.sampler_weight_to_count == 'linear':
            niters = optimizer.niters
            params = map(int, self.opt.sampler_linear_params.split(','))
            perc_a, perc_b, delay_a, delay_b = params
            epoch_iters = self.opt.epochs * self.num_samples / self.opt.batch_size
            perc = (1. * niters / epoch_iters) * (perc_b - perc_a) + perc_a
            delay = (1. * niters / epoch_iters) * (delay_b - delay_a) + delay_a
            ids = np.argsort(alpha)[:int(self.num_samples * perc / 100)]
            count = np.zeros(self.num_samples)
            count[ids] = int(delay)
            print('Delay perc: %d delay: %d' % (perc, delay))
        elif self.opt.sampler_weight_to_count == 'alpha_vs_dmom':
            # TODO: alpha_normed
            alpha = optimizer.alpha
            dmom = optimizer.alpha_mom_bc
            count = self.count
            ids = np.where(alpha[self.indices] < .01 * dmom[self.indices])[0]
            ids = self.indices[ids]
            count -= 1
            count[self.indices] = 0
            count[ids] = np.minimum(
                100, np.maximum(1, self.last_count[ids] * 2))
            self.last_count[ids] = count[ids]
        self.weights[self.indices] = 0
        # TODO: hyper param search
        self.weights = np.minimum(10, self.weights + 1)
        # self.count -= 1
        # self.count[self.indices] = count[self.indices]
        # self.count = np.maximum(0, self.count)
        # count = count-np.min(count)  # min should be 0
        # self.count = np.maximum(0, count)
        self.count = np.around(count - np.min(count))  # min should be 0
        while (self.count == 0).sum() < self.opt.batch_size:
            self.count = np.maximum(0, self.count - 1)
        self.visits[np.where(self.count == 0)[0]] += 1
        if self.opt.sampler_repetition:
            weights = np.ones_like(self.weights)
        else:
            weights = self.weights
        return weights

    def __len__(self):
        if self.training:
            return len(self.indices)
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
            n = num / num_class
            e = np.random.normal(0.0, 1.0, (dim, n))
            X[:, i * n:(i + 1) * n] = np.dot(D[:, :, i], e) + C[:, i:i + 1]
            Y[i * n:(i + 1) * n] = i
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        X = torch.Tensor(self.X[:, index]).float()
        Y = int(self.Y[index])
        return X, Y, index

    def __len__(self):
        return self.X.shape[1]


def get_logreg_loaders(opt, **kwargs):
    # np.random.seed(1234)
    np.random.seed(2222)
    # print("Create W")
    C = opt.c_const * random_orthogonal_matrix(1.0, (opt.dim, opt.num_class))
    D = opt.d_const * random_orthogonal_matrix(1.0,
                                               (opt.dim, opt.dim, opt.num_class))
    # print("Create train")
    trainset = IndexdLinearData(C, D, opt.num_train_data, opt.dim,
                                opt.num_class, train=True)
    # print("Create test")
    testset = IndexdLinearData(C, D, opt.num_test_data, opt.dim, opt.num_class,
                               train=False)
    torch.save((trainset.X, trainset.Y, testset.X, testset.Y,
                C), opt.logger_name + '/data.pth.tar')

    if opt.sampler:
        # weights = torch.ones(len(trainset)).double()
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size,
            # sampler=DmomWeightedRandomSampler(weights, len(trainset)),
            sampler=DelayedSampler(len(trainset), opt),
            **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=opt.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader
