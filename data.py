import torch
from torchvision import datasets, transforms


def get_loaders(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loaders(opt)
    elif opt.dataset == 'cifar10':
        return get_cifar10_loaders(opt)


class IndexedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super(IndexedMNIST, self).__getitem__(index)
        return img, target, index


def get_mnist_loaders(opt):
    kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        IndexedMNIST('data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
        batch_size=opt.batch_size, shuffle=True, **kwargs)
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
