import torch
import numpy as np
from torchvision import datasets
from torch.utils import data


class IndexedMNIST(datasets.MNIST):

    def __getitem__(self, index):
        img, target = super(IndexedMNIST, self).__getitem__(index)
        return img, target, index


class IndexedCIFAR10(datasets.CIFAR10):

    def __getitem__(self, index):
        img, target = super(IndexedCIFAR10, self).__getitem__(index)
        return img, target, index


class IndexedSVHN(datasets.SVHN):

    def __getitem__(self, index):
        img, target = super(IndexedSVHN, self).__getitem__(index)
        return img, target, index


class IndexedImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        img, target = super(IndexedImageFolder, self).__getitem__(index)
        return img, target, index


class IndexedMix(data.Dataset):

    def __init__(self, ds):
        super(IndexedMix, self).__init__()
        self.ds = ds
        self.lens = map(len, ds)

    def __getitem__(self, index):
        if index < self.lens[0]:
            img, target, _ = self.ds[0][index]
        else:
            img, target, _ = self.ds[1][index-self.lens[0]]
        return img, target, index

    def __len__(self):
        return sum(self.lens)


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
