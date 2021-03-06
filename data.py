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
    elif opt.dataset == 'cifar100':
        return get_cifar100_loaders(opt)
    elif opt.dataset == 'svhn':
        return get_svhn_loaders(opt)
    elif opt.dataset.startswith('imagenet'):
        return get_imagenet_loaders(opt)
    elif opt.dataset == 'logreg':
        return get_logreg_loaders(opt)
    elif opt.dataset == 'linreg':
        return get_linreg_loaders(opt)
    elif 'class' in opt.dataset:
        return get_logreg_loaders(opt)
    elif opt.dataset == 'rcv1':
        return get_rcv1_loaders(opt)
    elif opt.dataset == 'covtype':
        return get_covtype_loaders(opt)
    elif opt.dataset == 'protein':
        return get_protein_loaders(opt)
    elif opt.dataset == 'rf':
        return get_rf_loaders(opt)


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
    # if opt.g_estim == 'gluster':
    #     train_sampler = GlusterSampler(len(idxdataset), opt)
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


def get_gluster_loader(train_loader, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = train_loader.dataset
    if opt.g_imbalance:
        train_sampler = GlusterImbalanceSampler(len(idxdataset), opt)
    else:
        train_sampler = GlusterSampler(len(idxdataset), opt)
    raw_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.g_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True, **kwargs)
    return raw_loader, train_loader, train_sampler


def get_minvar_loader(train_loader, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = train_loader.dataset
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.g_batch_size,
        shuffle=True,
        drop_last=False, **kwargs)
    return train_loader


class IndexedDataset(data.Dataset):
    def __init__(self, dataset, opt, train=False, cr_labels=None):
        np.random.seed(2222)
        self.ds = dataset
        self.opt = opt

        # duplicates
        self.dup_num = 0
        self.dup_cnt = 0
        self.dup_ids = []
        self.imb_ids = []

        if opt.imbalance != '' and train:
            y = []
            for i in range(len(self.ds)):
                y += [self.ds[i][1]]
            y = np.array(y)
            params = map(float, self.opt.imbalance.split(','))
            self.imb_class, self.imb_ratio = params
            II = list(np.where(y == int(self.imb_class))[0])
            self.imb_ids = list(np.arange(len(self.ds)))
            self.imb_ids = list(np.delete(self.imb_ids, II))
            for i in range(int(self.imb_ratio)):
                self.imb_ids += II
            imb_left = (self.imb_ratio - int(self.imb_ratio))*len(II)
            self.imb_ids += list(np.array(II)[
                    np.random.permutation(len(II))[:int(imb_left)]])

        if opt.duplicate != '' and train:
            params = map(int, self.opt.duplicate.split(','))
            self.dup_num, self.dup_cnt = params
            self.dup_ids = np.random.permutation(len(dataset))[:self.dup_num]
            if isinstance(self.ds, LinearRegressionDataset):
                self.ds.dup_ids = self.dup_ids

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
        if len(self.imb_ids) > 0:
            img, target = self.ds[self.imb_ids[index]]
            return img, target, index
        subindex = index
        if self.opt.max_train_size > 0:
            subindex = subindex % self.opt.max_train_size
        if index >= len(self.ds):
            subindex = self.dup_ids[(index-len(self.ds))//self.dup_cnt]
        img, target = self.ds[subindex]
        if int(index) in self.cr_ids:
            # target = torch.tensor(self.cr_labels[index])
            target = self.cr_labels[index]
        return img, target, index

    def __len__(self):
        if len(self.imb_ids) > 0:
            return len(self.imb_ids)
        return len(self.ds)+self.dup_num*self.dup_cnt


def get_mnist_loaders(opt, **kwargs):
    transform = transforms.ToTensor()
    if not opt.no_transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST(
        opt.data, train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(opt.data, train=False, transform=transform)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_cifar10_100_transform(opt):
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
    return normalize, transform


def get_cifar10_loaders(opt):
    normalize, transform = get_cifar10_100_transform(opt)

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


def get_cifar100_loaders(opt):
    normalize, transform = get_cifar10_100_transform(opt)

    train_dataset = datasets.CIFAR100(root=opt.data, train=True,
                                      transform=transforms.Compose(transform),
                                      download=True)
    test_dataset = datasets.CIFAR100(
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
            II = self.scheduler.next_epoch()
            self.indices = II
            return (II[i] for i in torch.randperm(len(II)))
        return iter(torch.randperm(self.num_samples))

    def update(self):
        # self.scheduler.schedule()
        raise NotImplementedError("Should not be called.")

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
            II = self.indices
            # return (self.indices[i]
            #         for i in torch.randperm(len(self.indices)))
            if self.opt.sampler_repetition:
                W = self.weights[II]
                II = [np.ones(int(w), dtype=int)*i for i, w in zip(II, W)]
                II = np.concatenate(II)
            return (II[i] for i in torch.randperm(len(II)))
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
            II = np.where(self.count == 0)[0]
            self.visits[II] += self.weights[II]
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
            II = self.indices
            return (II[i] for i in torch.randperm(len(II)))
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


class InfiniteLoader(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_iter = iter([])
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            if isinstance(self.data_loader, list):
                II = self.data_loader
                self.data_iter = (II[i] for i in torch.randperm(len(II)))
            else:
                self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)
        return data

    def next(self):
        # for python2
        return self.__next__()

    def __len__(self):
        return len(self.data_loader)


class GlusterSampler(Sampler):
    def __init__(self, data_size, opt):
        self.data_size = data_size
        self.opt = opt
        self.assign_i = None
        self.cluster_size = None
        self.iters = None
        self.num_samples = opt.epoch_iters * opt.batch_size
        self.C = 0

    def set_assign_i(self, assign_i=None, cluster_size=None):
        if assign_i is None:
            self.assign_i = None
            return
        # self.assign_i = assign_i
        # self.cluster_size = cluster_size
        self.assign_i = np.zeros_like(assign_i)
        self.cluster_size = torch.zeros_like(cluster_size)
        self.iters = []

        # TODO: this justifies taking only g_optim_max steps
        # N = assign_i.shape[0]-59900
        # Nk = N//100
        # for i in range(100):
        #     self.iters += [iter(InfiniteLoader(range(i*Nk, (i+1)*Nk)))]
        #     self.cluster_size[i] = Nk
        # self.C = 101
        # self.iters += [iter(InfiniteLoader(range(N, assign_i.shape[0])))]
        # self.cluster_size[100] = 59900
        # return

        I0 = []
        for i in range(assign_i.max()+1):
            II = list(np.where(assign_i.flat == i)[0])
            if len(II) >= 1:
                self.cluster_size[len(self.iters)] = len(II)
                self.assign_i[II] = len(self.iters)
                self.iters += [iter(InfiniteLoader(II))]
            else:
                I0 += II
        C = len(self.iters)
        if len(I0) > 0:
            self.cluster_size[C] = len(I0)
            self.assign_i[I0] = C
            self.iters += [iter(InfiniteLoader(I0))]
            C += 1
        self.C = C

    def __iter__(self):
        if self.assign_i is None:
            for i in torch.randperm(self.data_size):
                yield i
            return

        incluster = self.opt.g_incluster
        if incluster >= 0:
            # only individual cluster variance
            for i in range(len(self.iters[incluster])):
                idx = next(self.iters[incluster])
                yield idx
            return

        C = self.C
        cur_c = 0
        cc = 0
        for i in range(self.num_samples):
            if cc == 0:
                cperm = torch.randperm(C)
            cur_c = cperm[cc]
            idx = next(self.iters[cur_c])
            cc = (cc+1) % len(self.iters)
            # cur_c = (cur_c+1) % len(self.iters)
            yield idx

    def __len__(self):
        if self.assign_i is None:
            return self.data_size
        return self.num_samples


class GlusterImbalanceSampler(GlusterSampler):
    def __init__(self, *args):
        super(GlusterImbalanceSampler, self).__init__(*args)

    def __iter__(self):
        # ***** NOT used
        # ***** NOT used
        # ***** NOT used
        # ***** NOT used
        # ***** NOT used
        # ***** NOT used
        # ***** NOT used
        C = len(self.iters)
        self.ratios = self.cluster_size[:C] / self.cluster_size[:C].min()
        if self.assign_i is None:
            for i in torch.randperm(self.data_size):
                yield i
            return

        cur_c = 0
        cur_j = 0
        for i in range(self.num_samples):
            idx = next(self.iters[cur_c])
            cur_j += 1
            if cur_j >= self.cluster_size[cur_c]:
                cur_c = (cur_c+1) % len(self.iters)
                cur_j = 0
            yield idx

    def __len__(self):
        if self.assign_i is None:
            return self.data_size
        return self.num_samples


def random_orthogonal_matrix(gain, shape, noortho=True):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                           "supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    if noortho:
        return a.reshape(shape)
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
            n = num // num_class
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


class PlainDataset(data.Dataset):
    def __init__(self, data, num_class=2, ratio=1, perm=None, xmean=None,
                 xstd=None):
        self.data = data
        if xmean is None:
            self.xmean = np.array(self.data[0].mean(0))
            E2x = self.xmean**2
            Ex2 = self.data[0].copy()
            Ex2.data **= 2
            Vx = Ex2.mean(0) - E2x
            self.xstd = np.array(np.sqrt(Vx))
            self.xmean, self.xstd = self.xmean.flatten(), self.xstd.flatten()
        else:
            self.xmean, self.xstd = xmean, xstd
        self.ymin = self.data[1].min()
        self.ymax = self.data[1].max()
        self.num_class = num_class
        self.classes = range(int(num_class))
        N = self.data[0].shape[0]
        # print(self.data[0].shape[1])
        # import ipdb; ipdb.set_trace()
        if perm is None:
            perm = np.random.permutation(N)
            self.ids = perm[:int(N*ratio)]
            self.no_ids = perm[int(N*ratio):]
        else:
            self.ids = perm

    def __getitem__(self, index):
        index = self.ids[index]
        xnorm = (self.data[0][index].toarray().flat
                 - self.xmean)/(self.xstd+1e-5)
        X = torch.Tensor(xnorm).float()
        Y = int((self.data[1][index]-self.ymin)/self.num_class)
        return X, Y

    def __len__(self):
        return len(self.ids)


class LibSVMDataset(PlainDataset):
    def __init__(self, fpath, *args, **kwargs):
        import sklearn
        import sklearn.datasets
        data = sklearn.datasets.load_svmlight_file(fpath)
        super(LibSVMDataset, self).__init__(data, *args, **kwargs)


class CSVDataset(PlainDataset):
    def __init__(self, fpath, *args, **kwargs):
        train_data = []
        train_labels = []
        import csv
        with open() as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                train_data += [float(x) for x in row[3:]]
                train_labels += [int(row[2])]
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        data = [train_data, train_labels]
        super(CSVDataset, self).__init__(data, *args, **kwargs)


def get_logreg_loaders(opt, **kwargs):
    # np.random.seed(1234)
    np.random.seed(2222)
    # print("Create W")
    C = opt.c_const * random_orthogonal_matrix(1.0, (opt.dim, opt.num_class),
                                               noortho=True)
    D = opt.d_const * random_orthogonal_matrix(
        1.0, (opt.dim, opt.dim, opt.num_class), noortho=True)
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


class LinearRegressionDataset(data.Dataset):

    def __init__(self, C, D, num, dim, num_class, r2=0, snr=1, w0=None,
                 train=True, online=False):
        (self.C, self.D, self.r2, self.snr, self.dim, self.num_class) = (
            C, D, r2, snr, dim, num_class)
        X = np.zeros((dim, num))
        Y = np.zeros((num_class, num))
        if w0 is None:
            w0 = np.random.normal(0, 1, (dim, num_class))
            w0 = w0/np.linalg.norm(w0)*r2
        self.w0 = w0
        for i in range(num_class):
            n = num // num_class
            Xcur, Ycur = self._gen_data(n)
            # X[:, i * n:(i + 1) * n] = np.dot(D[:, :, i], e) + C[:, i:i + 1]
            X[:, i * n:(i + 1) * n] = Xcur
            Y[:, i * n:(i + 1) * n] = Ycur
        self.X = X
        self.Y = Y
        self.classes = range(num_class)
        self.num_train = num
        self.dup_ids = None
        self.online = online

    def _gen_data(self, n):
        (C, D, r2, snr, dim, num_class) = (
            self.C, self.D, self.r2, self.snr, self.dim, self.num_class)
        w0 = self.w0
        s2 = 1
        if r2 > 1e-5:
            s2 = r2/snr
        ftype = 2
        if ftype == 1:
            e = np.random.normal(0.0, s2, (dim, n))
            X = D * e + C
            e = np.random.normal(0.0, .1, (num_class, n))
            Y = w0.T.dot(X) + e  # i + e
        else:
            X = np.random.normal(0.0, 1, (dim, n))
            e = np.random.normal(0.0, s2, (num_class, n))
            Y = w0.T.dot(X) + e
        return X, Y

    def __getitem__(self, index):
        if not self.online or (self.dup_ids is not None
                               and index in self.dup_ids):
            X, Y = self.X[:, index], self.Y[:, index]
        else:
            X, Y = self._gen_data(1)
            X, Y = X[:, 0], Y[:, 0]
        X = torch.Tensor(X).float()
        Y = torch.Tensor(Y).float()
        return X, Y

    def __len__(self):
        return self.num_train


def get_linreg_loaders(opt, **kwargs):
    # np.random.seed(1234)
    np.random.seed(2222)
    # print("Create W")
    # C = opt.c_const * random_orthogonal_matrix(1.0, (opt.dim, opt.num_class))
    # D = opt.d_const * random_orthogonal_matrix(
    #     1.0, (opt.dim, opt.dim, opt.num_class))
    # print("Create train")
    C = opt.c_const
    D = opt.d_const
    train_dataset = LinearRegressionDataset(
        C, D, opt.num_train_data, opt.dim, opt.num_class, r2=opt.r2,
        snr=opt.snr,
        train=True, online=opt.linreg_online)
    # print("Create test")
    test_dataset = LinearRegressionDataset(
        C, D, opt.num_test_data, opt.dim, opt.num_class, r2=opt.r2,
        snr=opt.snr, w0=train_dataset.w0,
        train=False)
    torch.save((train_dataset.X, train_dataset.Y,
                test_dataset.X, test_dataset.Y,
                C), opt.logger_name + '/data.pth.tar')

    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_rcv1_loaders(opt, **kwargs):
    # train_dataset = LibSVMDataset(
    #     os.path.join(opt.data, 'rcv1_train.binary.bz2'),
    #     num_class=opt.num_class, ratio=0.5)
    # perm = train_dataset.no_ids
    # test_dataset = LibSVMDataset(
    #     os.path.join(opt.data, 'rcv1_train.binary.bz2'),
    #     num_class=opt.num_class, perm=perm)
    train_dataset = LibSVMDataset(
        os.path.join(opt.data, 'rcv1_train.binary.bz2'),
        num_class=opt.num_class)
    xmean, xstd = train_dataset.xmean, train_dataset.xstd
    test_dataset = LibSVMDataset(
        # os.path.join(opt.data, 'rcv1_test.binary.bz2'),
        # xmean=xmean, xstd=xstd)
        os.path.join(opt.data, 'rcv1_train.binary.bz2'),
        xmean=xmean, xstd=xstd)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_covtype_loaders(opt, **kwargs):
    np.random.seed(2222)
    train_dataset = LibSVMDataset(
        os.path.join(opt.data, 'covtype.libsvm.binary.scale.bz2'),
        num_class=opt.num_class, ratio=0.5)
    xmean, xstd = train_dataset.xmean, train_dataset.xstd
    perm = train_dataset.no_ids
    test_dataset = LibSVMDataset(
        os.path.join(opt.data, 'covtype.libsvm.binary.scale.bz2'),
        num_class=opt.num_class, perm=perm, xmean=xmean, xstd=xstd)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_protein_loaders(opt, **kwargs):
    np.random.seed(2222)
    train_dataset = CSVDataset(
        os.path.join(opt.data, 'bio_train.dat'),
        num_class=opt.num_class, ratio=0.5)
    xmean, xstd = train_dataset.xmean, train_dataset.xstd
    perm = train_dataset.no_ids
    test_dataset = CSVDataset(
        os.path.join(opt.data, 'bio_train.dat'),
        num_class=opt.num_class, perm=perm, xmean=xmean, xstd=xstd)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


class RandomFeaturesDataset(data.Dataset):
    def __init__(self, num_train, dim, teacher, batch_size=256, ymean=None,
                 duplicate=''):
        import logging
        self.teacher = teacher
        self.ymean = ymean
        self.dim = list(dim)
        with torch.no_grad():
            self.X = torch.randn([num_train] + list(dim)).cuda()
            self.Y = torch.zeros([num_train]).cuda()
            # self.Y = teacher(self.X)[:, 0] > 0  # Only class 0 matters
            for i in range(0, num_train, batch_size):
                e = min(num_train, i+batch_size)
                self.Y[i:e] = teacher(self.X[i:e])[:, 0]
                logging.info('RF data: %d/%d' % (i, num_train))
            if ymean is None:
                self.ymean = self.Y.mean()
            self.Y = (self.Y-self.ymean) > 0
            if duplicate != '':
                dup_num, dup_ratio = map(float, duplicate.split(','))
                dup_total = int(num_train * dup_ratio)
                if dup_total > 0:
                    dup_ids = torch.LongTensor(
                        [x % dup_num for x in range(dup_total)]).cuda()
                    Xdup = torch.gather(
                        self.X, 0, dup_ids.view(-1, 1).expand(
                            (-1, self.X.shape[1])))
                    Ydup = torch.gather(self.Y, 0, dup_ids)
                    self.X = torch.cat((self.X[:-dup_total], Xdup), 0)
                    self.Y = torch.cat((self.Y[:-dup_total], Ydup), 0)

        self.X, self.Y = self.X.cpu(), self.Y.cpu()

    def __getitem__(self, index):
        return self.X[index, :].float(), self.Y[index].long()

    def __len__(self):
        return self.X.shape[0]


def get_rf_loaders(opt, **kwargs):
    # np.random.seed(2222)
    from models.rf import weight_reset
    if opt.teacher_arch == 'rf':
        from models.rf import RandomFeaturesModel
        teacher = RandomFeaturesModel(opt.dim, opt.teacher_hidden, 2).cuda()
        dim = (opt.dim,)
    elif opt.teacher_arch == 'resnet32':
        from models.cifar10 import resnet32
        teacher = resnet32(num_class=opt.num_class, nobatchnorm=False).cuda()
        with torch.no_grad():
            teacher.apply(weight_reset)
        teacher = torch.nn.DataParallel(teacher)
        dim = (3, 28, 28)
    train_dataset = RandomFeaturesDataset(opt.num_train_data, dim, teacher,
                                          duplicate=opt.duplicate)
    test_dataset = RandomFeaturesDataset(opt.num_test_data, dim, teacher,
                                         ymean=train_dataset.ymean)

    opt.duplicate = ''
    return dataset_to_loaders(train_dataset, test_dataset, opt)
