import unittest
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import sys
import time
import torch.optim as optim
from torchvision import datasets, transforms
import logging
sys.path.append('../')
from models.mnist import MLP, Convnet  # NOQA
from gluster.gluster import GradientClusterOnline  # NOQA
from gluster.gluster import GradientClusterBatch  # NOQA


def set_seed(seed):
    print('seed      : %d' % seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.set_printoptions(suppress=True)
    np.random.seed(seed)


def print_stats(model, gluster, X, T, batch_size):
    num = X.shape[0]
    W = list([param for name, param in model.named_parameters()])
    print(list([name for name, param in model.named_parameters()]))

    centers = gluster.get_centers()
    nclusters = centers[0].shape[0]

    dist = np.zeros((num, nclusters))
    normG = np.zeros((num,))
    normC = np.zeros((nclusters,))
    for i in range(num):
        x = Variable(X[i:i + 1]).cuda()
        t = Variable(T[i:i + 1]).cuda()
        model.zero_grad()
        y = model(x)
        # TODO: dist to centers for Conv is not zero
        # fix that, do all tests, move to sampling
        loss = F.nll_loss(y, t) / batch_size
        grad_params = torch.autograd.grad(loss, W)
        L = [(g - c.cuda()).pow(2).view(nclusters, -1).sum(1).cpu().numpy()
             for g, c in zip(grad_params, centers)]
        dist[i] = np.sum(L, 0)
        normG[i] = np.sum([g.pow(2).sum().item() for g in grad_params])
    L = [c.pow(2).view(nclusters, -1).sum(1).cpu().numpy()
         for c in centers]
    normC = np.sqrt(np.sum(L, 0))
    normG = np.sqrt(normG)
    dist = np.sqrt(dist)
    print('Dist:')
    print(dist)
    print('normG:')
    print(normG)
    print('normC:')
    print(normC)
    print('Reinit count: %s' % str(gluster.reinits.cpu().numpy()))
    print('Cluster size: %s' % str(gluster.cluster_size.cpu().numpy()))


def data_unique_n(train_size, nunique):
    X = torch.rand(train_size, 1, 28, 28)
    T = torch.ones(train_size).long()
    print('nunique   : %s' % str(nunique))

    for i in range(nunique, train_size, nunique):
        X[i:i + nunique].copy_(X[:nunique])
    Xte = X[:nunique]
    Yte = T[:nunique]
    return X, T, Xte, Yte


def data_unique_perc(train_size, punique):
    X = torch.rand(train_size, 1, 28, 28)
    T = torch.ones(train_size).long()
    print('punique   : %s' % str(punique))

    num = len(punique)
    sz = np.array(punique) * train_size - 1
    cur = num
    for i in range(num):
        X[cur:cur + int(sz[i])].copy_(X[i:i + 1])
        cur += int(sz[i])
    Xte = X[:len(punique)]
    Yte = T[:len(punique)]
    return X, T, Xte, Yte


def purturb_data(data, eps):
    X, T, Xte, Yte = data
    X += torch.rand(X.shape) * eps
    Xte += torch.rand(Xte.shape) * eps
    return X, T, Xte, Yte


class IndexedDataset():

    def __init__(self, dataset):
        self.ds = dataset

    def __getitem__(self, index):
        img, target = self.ds[index]
        return img, target, index

    def __len__(self):
        return len(self.ds)


def model_time(model, X, T, batch_size, niters):
    train_size = X.shape[0]
    model_tc = np.zeros(niters)
    for i in range(niters):
        tic = time.time()
        ids = np.random.permutation(train_size)[:batch_size]
        x = Variable(X[ids[:batch_size]]).cuda()
        t = Variable(T[ids[:batch_size]]).cuda()
        model.zero_grad()
        y = model(x)
        loss = F.nll_loss(y, t)
        loss.backward()
        toc = time.time()
        model_tc[i] = (toc - tic)
    return model_tc


def test_gluster_online(model, batch_size, data, nclusters, beta, min_size,
                        reinit_method, niters):
    X, T, Xte, Yte = data
    train_size = X.shape[0]
    print('batch_size: %d' % batch_size)
    print('nclusters : %d' % nclusters)
    print('beta      : %d' % beta)
    print('niters    : %d' % niters)
    print('train_size: %d' % train_size)
    print('min size  : %d' % min_size)
    print('reinit    : %s' % reinit_method)

    modelg = copy.deepcopy(model)
    gluster = GradientClusterOnline(modelg, beta, min_size, reinit_method,
                                    nclusters=nclusters)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    gluster_tc = np.zeros(niters)
    for i in range(niters):
        tic = time.time()
        ids = np.random.permutation(train_size)[:batch_size]
        x = Variable(X[ids[:batch_size]]).cuda()
        t = Variable(T[ids[:batch_size]]).cuda()
        modelg.zero_grad()
        # gluster.zero_data()
        y = modelg(x)
        loss = F.nll_loss(y, t)
        loss.backward()
        assign_i, batch_dist, _ = gluster.em_step()
        toc = time.time()
        gluster_tc[i] = (toc - tic)

    print('assign:')
    print(assign_i)
    # use model to prevent C update
    print_stats(model, gluster, Xte, Yte, batch_size)

    model_tc = model_time(model, X, T, batch_size, niters)

    print('Time:')
    print('%.4f +/- %.4f' % (gluster_tc.mean(), gluster_tc.std()))
    print('%.4f +/- %.4f' % (model_tc.mean(), model_tc.std()))
    # import ipdb; ipdb.set_trace()


class DataLoader(object):

    def __init__(self, X, T, batch_size):
        self.X = X
        self.T = T
        self.batch_size = batch_size

    def __iter__(self):
        X = self.X
        T = self.T
        batch_size = self.batch_size
        for i in range(0, X.shape[0], batch_size):
            a = i
            b = min(i + batch_size, X.shape[0])
            yield X[a:b], T[a:b], np.arange(a, b)

    def __len__(self):
        return self.X.shape[0]


def test_gluster_batch(
        model, batch_size, data, nclusters, min_size, citers):
    X, T, Xte, Yte = data
    train_size = X.shape[0]
    print('batch_size: %d' % batch_size)
    print('nclusters : %d' % nclusters)
    print('min_size  : %d' % min_size)
    print('citers    : %d' % citers)
    print('train_size: %d' % train_size)

    modelg = copy.deepcopy(model)
    gluster = GradientClusterBatch(modelg, min_size, nclusters=nclusters)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    gluster_tc = np.zeros(citers)
    data_loader = DataLoader(X, T, batch_size)
    for i in range(citers):
        tic = time.time()
        gluster.update_batch(data_loader, train_size)
        toc = time.time()
        gluster_tc[i] = (toc - tic)

    # get test data assignments
    gluster.eval()  # gluster test mode
    x = Variable(Xte).cuda()
    t = Variable(Yte).cuda()
    modelg.zero_grad()
    # gluster.zero_data()
    y = modelg(x)
    loss = F.nll_loss(y, t)
    loss.backward()
    assign_i = gluster.assign()

    print('assign:')
    print(assign_i)
    # use model to prevent C update
    print_stats(model, gluster, Xte, Yte, batch_size)

    tic = time.time()
    model_tc = model_time(model, X, T, batch_size, citers)
    toc = time.time()

    print('Time:')
    print('%.4f +/- %.4f' % (gluster_tc.mean(), gluster_tc.std()))
    print('%.4f +/- %.4f' % (model_tc.mean(), model_tc.std()))
    print('%.4f' % (toc - tic))
    # import ipdb; ipdb.set_trace()


def train(epoch, train_loader, model, optimizer):
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        model.train()
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        # gluster.zero()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t Loss: {loss:.6f}\t'.format(
                epoch, batch_idx, len(train_loader),
                loss=loss.item()))


def test_mnist(
        model, batch_size, epochs, nclusters, min_size, citers, figname,
        ignore_modules=[], no_grad=False):
    print('batch_size: %d' % batch_size)
    print('epochs    : %d' % epochs)
    print('nclusters : %d' % nclusters)
    print('min_size  : %d' % min_size)
    print('citers    : %d' % citers)

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

    optimizer = optim.SGD(model.parameters(),
                          lr=.01, momentum=0.9,
                          weight_decay=.0005)

    for e in range(epochs):
        train(e, train_loader, model, optimizer)

    modelg = copy.deepcopy(model)
    # model's weight are not going to change, opt.step() is not called
    gluster = GradientClusterBatch(modelg, min_size, nclusters=nclusters,
                                   ignore_modules=ignore_modules,
                                   no_grad=no_grad)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    gluster_tc = np.zeros(citers)
    total_dist = float('inf')
    pred_i = 0
    loss_i = 0
    for i in range(citers):
        tic = time.time()
        stat = gluster.update_batch(train_loader, len(train_dataset))
        if i > 0:
            assert pred_i.sum() == stat[3].sum(), 'predictions changed'
            assert loss_i.sum() == stat[4].sum(), 'loss changed'
            assert stat[0].sum() <= total_dist.sum(), 'Total dists went up'
        total_dist, assign_i, target_i, pred_i, loss_i, topk_i = stat
        toc = time.time()
        gluster_tc[i] = (toc - tic)
        normC = gluster.print_stats()

    print('%.4f +/- %.4f' % (gluster_tc.mean(), gluster_tc.std()))
    torch.save({'assign': assign_i, 'target': target_i,
                'pred': pred_i, 'loss': loss_i,
                'normC': normC},
               figname)
    # import ipdb; ipdb.set_trace()


def test_mnist_online(
        model, batch_size, epochs, nclusters, beta, min_size,
        reinit_method, figname):
    print('batch_size: %d' % batch_size)
    print('epochs    : %d' % epochs)
    print('nclusters : %d' % nclusters)
    print('beta      : %d' % beta)
    print('min_size  : %d' % min_size)
    print('reinit    : %s' % reinit_method)

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
    gluster = GradientClusterOnline(modelg, beta, min_size,
                                    reinit_method, nclusters=nclusters)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    optimizer = optim.SGD(modelg.parameters(),
                          lr=.01, momentum=0.9,
                          weight_decay=.0005)

    train_size = len(train_dataset)
    gluster_tc = []
    assign_i = -np.ones((train_size, 1))
    pred_i = np.zeros(train_size)
    loss_i = np.zeros(train_size)
    target_i = np.zeros(train_size)
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
            pred_i[idx] = output.max(1, keepdim=True)[1].cpu().numpy()[:, 0]
            target_i[idx] = target.cpu().numpy()
            loss_i[idx] = loss.detach().cpu().numpy()
            loss = loss.mean()
            loss.backward()
            ai, batch_dist, iv = gluster.em_step()
            ai = ai.cpu().numpy()
            assign_i[idx] = ai
            # TODO: multiple iv
            if len(iv) > 0:
                iv[0] = iv[0].cpu().numpy()
                assign_i[assign_i == iv[0]] = -1
                ai[ai == iv[0]] = -1
            # optim
            optimizer.step()
            toc = time.time()
            gluster_tc += [toc - tic]
            if batch_idx % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t Loss: {loss:.6f}\t'.format(
                    e, batch_idx, len(train_loader),
                    loss=loss.item()))
                print('assign:')
                print(ai[:10])
                print(batch_dist[:10])
                normC = gluster.print_stats()
                print('%.4f +/- %.4f' % (np.mean(gluster_tc),
                                         np.std(gluster_tc)))
                # import ipdb; ipdb.set_trace()

    u, c = np.unique(assign_i, return_counts=True)
    print(u)
    print(c)
    torch.save({'assign': assign_i, 'target': target_i,
                'pred': pred_i, 'loss': loss_i,
                'normC': normC},
               figname)


def test_mnist_online_delayed(
        model, batch_size, epochs, nclusters, beta, min_size,
        reinit_method, delay, figname):
    print('batch_size: %d' % batch_size)
    print('epochs    : %d' % epochs)
    print('nclusters : %d' % nclusters)
    print('beta      : %d' % beta)
    print('min_size  : %d' % min_size)
    print('reinit    : %s' % reinit_method)
    print('delay     : %d' % delay)

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

    model = MLP(dropout=False)
    model.cuda()
    modelg = copy.deepcopy(model)
    gluster = GradientClusterOnline(modelg, beta, min_size,
                                    reinit_method, nclusters=nclusters)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    optimizer = optim.SGD(model.parameters(),
                          lr=.01, momentum=0.9,
                          weight_decay=.0005)

    train_size = len(train_dataset)
    gluster_tc = []
    train_tc = []
    niter = 0
    for e in range(epochs):
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            tic = time.time()
            model.train()
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            # optim
            optimizer.step()
            toc = time.time()
            train_tc += [toc - tic]
            niter += 1
            if niter % delay == 0:
                tic = time.time()
                modelg.train()
                gluster.copy_(model)
                # modelg.zero_grad()
                # gluster.zero_data()
                output = modelg(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                ret = gluster.em_step()
                toc = time.time()
                gluster_tc += [toc - tic]
                if ret is not None:
                    ai, batch_dist, _ = ret
                    print('assign:')
                    print(ai[:10])
                    print(batch_dist[:10])
                    normC = gluster.print_stats()
                    print('%.4f +/- %.4f' % (np.mean(gluster_tc),
                                             np.std(gluster_tc)))
                    # import ipdb; ipdb.set_trace()
            if batch_idx % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t Loss: {loss:.6f}\t'.format(
                    e, batch_idx, len(train_loader),
                    loss=loss.item()))

    assign_i = np.zeros((train_size, 1)) - 1
    pred_i = np.zeros(train_size)
    loss_i = np.zeros(train_size)
    target_i = np.zeros(train_size)
    gluster.eval()
    gluster.copy_(model)
    normC = gluster.print_stats()
    gluster_eval_tc = []
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        tic = time.time()
        data, target = data.cuda(), target.cuda()
        modelg.zero_grad()
        # gluster.zero_data()
        output = modelg(data)
        loss = F.nll_loss(output, target, reduction='none')
        # store stats
        pred_i[idx] = output.max(1, keepdim=True)[1].cpu().numpy()[:, 0]
        target_i[idx] = target.cpu().numpy()
        loss_i[idx] = loss.detach().cpu().numpy()
        loss = loss.mean()
        loss.backward()
        ai, batch_dist = gluster.em_step()
        assign_i[idx] = ai.cpu().numpy()
        toc = time.time()
        gluster_eval_tc += [toc - tic]
        if batch_idx % 10 == 0:
            u, c = np.unique(assign_i, return_counts=True)
            print('Assignment: [{0}/{1}]: {2}\t{3}'.format(
                batch_idx, len(train_loader), str(list(u)), str(list(c))))
    u, c = np.unique(assign_i, return_counts=True)
    print(u)
    print(c)
    # not counting data prep time
    print('Train time: %.4fs' % np.sum(train_tc))
    print('Gluster update time: %.4fs' % np.sum(gluster_tc))
    print('Gluster eval time: %.4fs' % np.sum(gluster_eval_tc))
    torch.save({'assign': assign_i, 'target': target_i,
                'pred': pred_i, 'loss': loss_i,
                'normC': normC},
               figname)


class ToyTests(object):
    def test_few_iters(self):
        model = self.model
        # Few iterations
        data = data_unique_n(100, 5)
        test_gluster_online(model, 10, data, 5, .9, 1, 'data', 10)
        print(">>> One cluster still hasn't converged?")
        test_gluster_online(model, 10, data, 5, .9, 1, 'largest', 10)
        print(">>> But the init with largest takes longer")

    def test_more_iters(self):
        model = self.model
        # More iterations
        data = data_unique_n(100, 5)
        test_gluster_online(model, 10, data, 5, .9, 1, 'data', 100)
        print(">>> This has now converged.")
        test_gluster_online(model, 10, data, 5, .9, 1, 'largest', 100)
        print(">>> centers should match input based on Dist")

    def test_more_data(self):
        model = self.model
        # More unique data than centers
        data = data_unique_n(100, 10)
        test_gluster_online(model, 10, data, 5, .9, 1, 'data', 100)
        test_gluster_online(model, 10, data, 5, .9, 1, 'largest', 100)
        print(
                "*** Clusters should get about the same "
                "number of data points, but they don't, why?***")

    def test_more_centers(self):
        model = self.model
        # More centers than data
        # TODO: stop reinit if more centers
        data = data_unique_n(100, 5)
        test_gluster_online(model, 10, data, 10, .9, 1, 'data', 100)
        print(
                "*** 5 centers are reinited > 30 "
                "times but first 5 are stable ***")
        test_gluster_online(model, 10, data, 10, .9, 1, 'largest', 100)

    def test_imbalance(self):
        model = self.model
        # Imbalance data
        data = data_unique_perc(100, [.9, .1])
        test_gluster_online(model, 10, data, 2, .9, 1, 'data', 10)
        test_gluster_online(model, 10, data, 2, .9, 1, 'largest', 10)
        print(
                "***  finds 2 clusters in 1 reinit "
                "but hasn't converged in 10 ***")

    def test_imbalance_more_iters(self):
        model = self.model
        # More iterations
        data = data_unique_perc(100, [.9, .1])
        test_gluster_online(model, 10, data, 2, .9, 1, 'data', 100)
        test_gluster_online(model, 10, data, 2, .9, 1, 'largest', 100)
        print("***  converged now ***")

    def test_time(self):
        model = self.model
        # Time test
        data = data_unique_perc(1000, [.9, .1])
        test_gluster_online(model, 128, data, 2, .9, 1, 'data', 100)
        print("2-3x solwer")

    def test_noise(self):
        model = self.model
        # noise
        data = data_unique_perc(100, [.9, .1])
        data = purturb_data(data, .01)
        test_gluster_online(model, 10, data, 2, .9, 1, 'data', 100)
        test_gluster_online(model, 10, data, 2, .9, 1, 'largest', 100)

    def test_gluster_batch(self):
        model = self.model
        # gluster batch
        data = data_unique_n(100, 5)
        test_gluster_batch(model, 10, data, 5, 1, 10)

    def test_gluster_batch_noise(self):
        model = self.model
        # gluster batch noise
        data = data_unique_perc(100, [.9, .1])
        data = purturb_data(data, .01)
        # TODO: seed 12345
        test_gluster_batch(model, 10, data, 2, 1, 10)


class MNISTTest(object):
    def test_mnist_citer2(self):
        model = self.model
        # MNIST
        citers = 2
        figname = self.prefix+',nclusters_2,citers_2.pth.tar'
        test_mnist(model, 128, 2, 2, 10, citers, figname)

    def test_mnist_citer10(self):
        model = self.model
        citers = 10
        figname = self.prefix+',nclusters_2,citers_10.pth.tar'
        test_mnist(model, 128, 2, 2, 10, citers, figname)

    def test_mnist_nclusters10(self):
        model = self.model
        nclusters = 10
        citers = 10
        figname = self.prefix+',nclusters_10,citers_10.pth.tar'
        test_mnist(model, 128, 2, nclusters, 10, citers, figname)

    def test_mnist_online(self):
        model = self.model
        # Online gluster on MNIST
        epochs = 2
        nclusters = 10
        beta = .999
        min_size = 1
        reinit_method = 'largest'
        figname = self.prefix+',nclusters_10,online.pth.tar'
        test_mnist_online(model, 128, epochs, nclusters,
                          beta, min_size, reinit_method, figname)

    def test_mnist_online_delayed(self):
        model = self.model
        # Online gluster with delayed update
        epochs = 2
        nclusters = 10
        beta = .999
        min_size = 1
        reinit_method = 'largest'
        delay = 10
        figname = (
            self.prefix+',nclusters_10,online,delay_10.pth.tar')
        test_mnist_online_delayed(
                model,
                128, epochs, nclusters, beta, min_size, reinit_method, delay,
                figname)

    def test_mnist_online_delay100(self):
        model = self.model
        epochs = 2
        nclusters = 10
        beta = .999
        min_size = 1
        reinit_method = 'largest'
        delay = 100
        figname = (
            self.prefix+',nclusters_10,online,delay_100.pth.tar')
        test_mnist_online_delayed(
                model,
                128, epochs, nclusters, beta, min_size, reinit_method, delay,
                figname)

    def test_mnist_input(self):
        model = self.model
        # Batch gluster layer 1, input only
        nclusters = 10
        citers = 10
        no_grad = True
        ignore_modules = ['fc2', 'fc3', 'fc4']
        figname = self.prefix+',nclusters_10,input.pth.tar'
        test_mnist(
                model, 128, 2, nclusters, 10, citers, figname,
                ignore_modules, no_grad=no_grad)

    def test_mnist_layer1(self):
        model = self.model
        # Batch gluster layer 1
        nclusters = 10
        citers = 10
        ignore_modules = ['fc2', 'fc3', 'fc4']
        figname = self.prefix+',nclusters_10,layer_1.pth.tar'
        test_mnist(
                model, 128, 2, nclusters, 10, citers,
                figname, ignore_modules)

    def test_mnist_layer4(self):
        model = self.model
        # Batch gluster layer 4
        nclusters = 10
        citers = 10
        ignore_modules = ['fc1', 'fc2', 'fc3']
        figname = self.prefix+',nclusters_10,layer_4.pth.tar'
        test_mnist(model, 128, 2, nclusters, 10,
                   citers, figname, ignore_modules)


class TestGlusterMLP(unittest.TestCase, ToyTests, MNISTTest):
    def setUp(self):
        print('In method %s' % self._testMethodName)
        logging.basicConfig(
                format='%(asctime)s %(message)s', level=logging.INFO)
        set_seed(1234)
        self.model = MLP(dropout=False)
        self.model.cuda()
        print('Model initialized.')
        self.prefix = '/u/faghri/dmom/code/notebooks/figs_gluster/mlp'


class TestGlusterConv(unittest.TestCase, ToyTests, MNISTTest):
    def setUp(self):
        print('In method %s' % self._testMethodName)
        logging.basicConfig(
                format='%(asctime)s %(message)s', level=logging.INFO)
        set_seed(1234)
        self.model = Convnet(dropout=False)
        self.model.cuda()
        print('Model initialized.')
        self.prefix = '/u/faghri/dmom/code/notebooks/figs_gluster/cnn'


if __name__ == '__main__':
    unittest.main()
