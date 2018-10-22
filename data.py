import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import schedulers


def get_loaders(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loaders(opt)
    elif opt.dataset == 'cifar10':
        return get_cifar10_loaders(opt)
    elif opt.dataset == 'svhn':
        return get_svhn_loaders(opt)
    elif opt.dataset.startswith('imagenet'):
        return get_imagenet_loaders(opt)
    elif opt.dataset == 'logreg':
        return get_logreg_loaders(opt)
    elif 'class' in opt.dataset:
        return get_logreg_loaders(opt)


def dataset_to_loaders(train_dataset, test_dataset, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = IndexedDataset(train_dataset, opt, train=True)
    if opt.sampler:
        if opt.scheduler:
            train_sampler = DMomSampler(len(idxdataset), opt)
        elif opt.minvar:
            train_sampler = MinVarSampler(len(idxdataset), opt)
        else:
            train_sampler = DelayedSampler(len(idxdataset), opt)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(test_dataset, opt),
        batch_size=opt.test_batch_size, shuffle=False,
        **kwargs)

    train_test_loader = torch.utils.data.DataLoader(
        IndexedDataset(train_dataset, opt, train=True,
                       cr_labels=idxdataset.cr_labels),
        batch_size=opt.test_batch_size, shuffle=False,
        **kwargs)
    return train_loader, test_loader, train_test_loader


class IndexedDataset(data.Dataset):
    def __init__(self, dataset, opt, train=False, cr_labels=None):
        np.random.seed(2222)
        self.ds = dataset
        self.opt = opt

        # duplicates
        self.dup_num = 0
        self.dup_cnt = 0
        self.dup_ids = []
        if opt.duplicate != '' and train:
            params = map(int, self.opt.duplicate.split(','))
            self.dup_num, self.dup_cnt = params
            self.dup_ids = np.random.permutation(len(dataset))[:self.dup_num]

        # corrupt labels
        if cr_labels is not None:
            self.cr_labels = cr_labels
        else:
            self.cr_labels = np.random.randint(
                self.opt.num_class, size=len(self))
        cr_ids = np.arange(len(self))
        self.cr_ids = []
        if train:
            cr_num = int(1. * opt.corrupt_perc * len(dataset) / 100.)
            self.cr_ids = cr_ids[:cr_num]

    def __getitem__(self, index):
        subindex = index
        if index >= len(self.ds):
            subindex = self.dup_ids[(index-len(self.ds))/self.dup_cnt]
        img, target = self.ds[subindex]
        if index in self.cr_ids:
            target = torch.tensor(self.cr_labels[index])
        return img, target, index

    def __len__(self):
        return len(self.ds)+self.dup_num*self.dup_cnt


def get_mnist_loaders(opt, **kwargs):
    train_dataset = datasets.MNIST(
        opt.data, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    test_dataset = datasets.MNIST(
            opt.data, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_cifar10_loaders(opt):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2023, 0.1994, 0.2010))

    # valid_size=0.1
    # split = int(np.floor(valid_size * num_train))
    # indices = list(range(num_train))
    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    if opt.data_aug:
        transform = [
            transforms.RandomAffine(10, (.1, .1), (0.7, 1.2), 10),
            transforms.ColorJitter(.2, .2, .2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    train_dataset = datasets.CIFAR10(root=opt.data, train=True,
                                     transform=transforms.Compose(transform),
                                     download=True)
    test_dataset = datasets.CIFAR10(
        root=opt.data, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_svhn_loaders(opt, **kwargs):
    normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))
    if opt.data_aug:
        transform = [
            transforms.RandomAffine(10, (.1, .1), (0.7, 1.), 10),
            transforms.ColorJitter(.2, .2, .2),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]

    train_dataset = torch.utils.data.ConcatDataset(
        (datasets.SVHN(
            opt.data, split='train', download=True,
            transform=transforms.Compose(transform)),
         datasets.SVHN(
             opt.data, split='extra', download=True,
             transform=transforms.Compose(transform))))
    test_dataset = datasets.SVHN(opt.data, split='test', download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))
                                 ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_imagenet_loaders(opt):
    # Data loading code
    traindir = os.path.join(opt.data, 'train')
    valdir = os.path.join(opt.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


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


class DMomSampler(Sampler):
    def __init__(self, num_samples, opt):
        self.num_samples = num_samples
        self.opt = opt
        self.training = True
        self.scheduler = schedulers.get_scheduler(num_samples, opt)

    def __iter__(self):
        if self.training:
            I = self.scheduler.next_epoch()
            self.indices = I
            return (I[i] for i in torch.randperm(len(I)))
        return iter(torch.randperm(self.num_samples))

    def update(self):
        # self.scheduler.schedule()
        raise NotImplemented("Should not be called.")

    def __len__(self):
        if self.training:
            return len(self.indices)
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
        params = map(int, self.opt.sampler_params.split(','))
        self.perc = params[0]
        self.rank = np.ones(num_samples)*num_samples

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
        alpha = np.array(alpha_val, dtype=np.float32)

        if self.opt.sampler_w2c == '1_over':
            amx = 1. / self.opt.sampler_max_count
            count = 1. / np.maximum(amx, alpha)
            # count = count-np.min(count)  # min should be 0
            # weights = 1./(count+1)
            # weights = count+1
        elif self.opt.sampler_w2c == 'log':
            amx = np.exp(-self.opt.sampler_max_count)
            count = -np.log(np.maximum(amx, alpha))
            # count = count-np.min(count)  # min should be 0
            # weights = np.exp(-count)
            # weights = count+1
        elif self.opt.sampler_w2c == 'c_times':
            amx = self.opt.sampler_max_count
            alpha = np.minimum(1, np.maximum(0, alpha))  # ensuring in [0,1]
            count = amx * (1 - alpha)
        elif self.opt.sampler_w2c == 'linear':
            niters = optimizer.niters
            params = map(int, self.opt.sampler_params.split(','))
            perc_a, perc_b, delay_a, delay_b = params
            epoch_iters = (self.opt.epochs * self.num_samples
                           / self.opt.batch_size)
            perc = (1. * niters / epoch_iters) * (perc_b - perc_a) + perc_a
            delay = (1. * niters / epoch_iters) * (delay_b - delay_a) + delay_a
            ids = np.argsort(alpha)[:int(self.num_samples * perc / 100)]
            count = np.zeros(self.num_samples)
            count[ids] = int(delay)
            print('Delay perc: %d delay: %d' % (perc, delay))
        elif self.opt.sampler_w2c == 'alpha_vs_dmom':
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
        elif self.opt.sampler_w2c == 'linear2':
            niters = optimizer.niters
            params = map(int, self.opt.sampler_params.split(','))
            perc_a, perc_b, delay_a, delay_b = params
            epoch_iters = (self.opt.epochs * self.num_samples
                           / self.opt.batch_size)
            perc = (1. * niters / epoch_iters) * (perc_b - perc_a) + perc_a
            delay = (1. * niters / epoch_iters) * (delay_b - delay_a) + delay_a
            ids = np.arange(len(alpha))
            np.random.shuffle(ids)
            ids = ids[np.argsort(alpha[ids])[
                :int(self.num_samples * perc / 100)]]
            count = np.zeros(self.num_samples)
            count[ids] = int(delay)
            nonvisited = (self.count != 0)
            count[nonvisited] = self.count[nonvisited]-1
            print('Delay perc: %d delay: %d' % (perc, delay))
        elif self.opt.sampler_w2c == 'linrank':
            niters = optimizer.niters
            params = map(int, self.opt.sampler_params.split(','))
            perc_a, perc_b, delay_a, delay_b = params
            epoch_iters = (self.opt.epochs * self.num_samples
                           / self.opt.batch_size)
            perc = (1. * niters / epoch_iters) * (perc_b - perc_a) + perc_a
            count = self.count
            visited = (count == 0)
            revisits = (count == 1)

            ids = np.arange(len(alpha))
            np.random.shuffle(ids)
            rank = np.zeros(len(alpha))
            alpha_x = np.array(alpha)
            alpha_x[revisits] = alpha.max() + 1
            srti = np.argsort(alpha_x[ids])
            rank[ids[srti]] = np.arange(len(alpha))

            drop_rank = int(self.num_samples * perc / 100)
            ids_d = ids[srti[:drop_rank]]
            ids_r = ids[srti[drop_rank:]]
            tobevisited = np.zeros(self.num_samples, dtype=np.bool)
            tobevisited[ids_r] = True
            count[count > 1] -= 1  # not the revisits that won't be visited
            # count[:] = 3
            count[tobevisited] = 0

            needdelay = np.logical_and(visited, np.logical_not(tobevisited))
            delay = (drop_rank-rank[needdelay])/drop_rank * niters/epoch_iters
            delay = (1. * delay) * (delay_b - delay_a) + delay_a
            count[needdelay] = np.int32(np.ceil(delay))
            print('Delay perc: %d delay: %d' % (perc, count[ids_d].mean()))
        elif self.opt.sampler_w2c == 'autoexp':
            # assume we have dropped p% and revisited k items
            # last_count > 0 for those k items, their ids are in self.indices
            # 1. decide if any of these k items have to be revisited
            #    1.1. compute rank, if rank is higher than p
            # 2. use p-k+pi for next step and include revisit items
            niters = optimizer.niters
            params = map(int, self.opt.sampler_params.split(','))
            perc_a, perc_b, delay_a, delay_b = params
            iperc = 1. * (perc_b - perc_a) / self.opt.epochs
            epoch_iters = (self.opt.epochs * self.num_samples
                           / self.opt.batch_size)
            delay = (1. * niters / epoch_iters) * (delay_b - delay_a) + delay_a
            perc = min(perc_b, self.perc + iperc)
            # sort and rank
            ids = np.arange(len(alpha))
            np.random.shuffle(ids)
            rank = np.zeros(len(alpha))
            srti = np.argsort(alpha[ids])
            rank[ids[srti]] = np.arange(len(alpha))
            # test:
            # A=np.zeros(len(alpha))
            # A[np.int32(rank)]=alpha
            nonvisited = (self.count != 0)
            visited = (self.count == 0)
            bumped = np.logical_and(visited, rank > self.rank)
            perc -= bumped.sum()/self.num_samples*100
            self.perc = perc
            self.rank[visited] = rank[visited]
            ids = ids[srti[:int(self.num_samples * perc / 100)]]
            count = np.zeros(self.num_samples)
            count[ids] = np.minimum(
                delay, np.maximum(1, self.last_count[ids] * 2))
            count[bumped] = 0
            self.last_count[ids] = count[ids]
            self.last_count[bumped] = count[bumped]
            count[nonvisited] = self.count[nonvisited]-1
            print('Delay perc: %d delay: %d' % (perc, delay))
        self.weights[self.indices] = 0
        # TODO: hyper param search
        self.weights = np.minimum(self.opt.sampler_maxw, self.weights + 1)
        # self.count -= 1
        # self.count[self.indices] = count[self.indices]
        # self.count = np.maximum(0, self.count)
        # count = count-np.min(count)  # min should be 0
        # self.count = np.maximum(0, count)
        self.count = np.around(count - np.min(count))  # min should be 0
        while (self.count == 0).sum() < self.opt.batch_size:
            self.count = np.maximum(0, self.count - 1)
        if self.opt.sampler_repetition:
            weights = np.ones_like(self.weights)
            I = np.where(self.count == 0)[0]
            self.visits[I] += self.weights[I]
        else:
            weights = self.weights
            self.visits[np.where(self.count == 0)[0]] += 1
        # print(np.histogram(self.visits, bins=10))
        # print(alpha.min())
        # print(alpha.max())
        # import ipdb; ipdb.set_trace()
        count_nz = float((count == 0).sum())
        self.logger.update('touch_p', count_nz*100./len(count), 1)
        self.logger.update('touch', count_nz, 1)
        self.logger.update('count_h', count, 1, hist=True)
        self.logger.update('weights_h', self.weights, 1, hist=True)
        self.logger.update('visits_h', self.visits, 1, hist=True)
        return weights

    def __len__(self):
        if self.training:
            return len(self.indices)
        return self.num_samples


class MinVarSampler(Sampler):
    def __init__(self, train_size, opt):
        self.delta = np.zeros(train_size)
        self.wait = np.zeros(train_size)
        self.active = np.ones(train_size, dtype=np.bool)
        self.count = np.zeros(train_size)
        self.weights = np.ones(train_size)
        self.indices = range(train_size)
        self.training = True
        self.opt = opt
        self.num_samples = train_size
        self.visits = np.zeros(train_size)

    def __iter__(self):
        if self.training:
            self.indices = np.where(self.count == 0)[0]
            I = self.indices
            return (I[i] for i in torch.randperm(len(I)))
        return iter(torch.randperm(self.num_samples))

    def update(self, optimizer):
        alpha_val = optimizer.grad_norm
        lmbd = float(self.opt.sampler_params)
        alpha = np.array(alpha_val, dtype=np.float32)
        alpha = alpha.clip(1e-20)
        self.delta[self.active] = .5*lmbd/alpha[self.active]
        self.delta.clip(0, self.opt.sampler_maxw-1)
        self.weights[self.active] = self.delta[self.active]+1
        # self.weights[self.active] = 1

        # shared updates
        self.active[:] = False
        while self.active.sum() < self.opt.min_batches * self.opt.batch_size:
            self.wait[np.logical_not(self.active)] += 1
            self.active[self.delta <= self.wait] = True
        self.count[:] = (self.delta - self.wait).clip(0)
        self.visits[self.active] += 1
        self.wait[self.active] = 0
        na = (1-self.active).sum()
        count_nz = float((self.count == 0).sum())
        self.logger.update('snooze', float(na), 1)
        self.logger.update('snooze_p', float(na*100./self.num_samples), 1)
        self.logger.update('touch_p', count_nz*100./len(self.count), 1)
        self.logger.update('touch', count_nz, 1)
        self.logger.update('count_h', self.count, 1, hist=True)
        self.logger.update('weights_h', self.weights, 1, hist=True)
        self.logger.update('visits_h', self.visits, 1, hist=True)
        return self.weights

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


class LinearDataset(data.Dataset):

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
        self.classes = range(num_class)

    def __getitem__(self, index):
        X = torch.Tensor(self.X[:, index]).float()
        Y = int(self.Y[index])
        return X, Y

    def __len__(self):
        return self.X.shape[1]


def get_logreg_loaders(opt, **kwargs):
    # np.random.seed(1234)
    np.random.seed(2222)
    # print("Create W")
    C = opt.c_const * random_orthogonal_matrix(1.0, (opt.dim, opt.num_class))
    D = opt.d_const * random_orthogonal_matrix(
        1.0, (opt.dim, opt.dim, opt.num_class))
    # print("Create train")
    train_dataset = LinearDataset(C, D, opt.num_train_data, opt.dim,
                                  opt.num_class, train=True)
    # print("Create test")
    test_dataset = LinearDataset(C, D,
                                 opt.num_test_data, opt.dim, opt.num_class,
                                 train=False)
    torch.save((train_dataset.X, train_dataset.Y,
                test_dataset.X, test_dataset.Y,
                C), opt.logger_name + '/data.pth.tar')

    return dataset_to_loaders(train_dataset, test_dataset, opt)
