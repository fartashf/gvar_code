import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import sys
import time
import torch.optim as optim
from torchvision import datasets, transforms
sys.path.append('../')
from models.mnist import MLP, MNISTNet  # NOQA
from gluster import GradientClusterOnline  # NOQA
from gluster import GradientClusterBatch  # NOQA


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
        x = Variable(X[i:i+1]).cuda()
        t = Variable(T[i:i+1]).cuda()
        model.zero_grad()
        y = model(x)
        loss = F.nll_loss(y, t)/batch_size
        grad_params = torch.autograd.grad(loss, W)
        L = [(g-c).pow(2).view(nclusters, -1).sum(1).cpu().numpy()
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
        X[i:i+nunique].copy_(X[:nunique])
    Xte = X[:nunique]
    Yte = T[:nunique]
    return X, T, Xte, Yte


def data_unique_perc(train_size, punique):
    X = torch.rand(train_size, 1, 28, 28)
    T = torch.ones(train_size).long()
    print('punique   : %s' % str(punique))

    num = len(punique)
    sz = np.array(punique)*train_size - 1
    cur = num
    for i in range(num):
        X[cur:cur+int(sz[i])].copy_(X[i:i+1])
        cur += int(sz[i])
    Xte = X[:len(punique)]
    Yte = T[:len(punique)]
    return X, T, Xte, Yte


def purturb_data(data, eps):
    X, T, Xte, Yte = data
    X += torch.rand(X.shape)*eps
    Xte += torch.rand(Xte.shape)*eps
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
        model_tc[i] = (toc-tic)
    return model_tc


def test_gluster_online(batch_size, data, nclusters, beta, seed, niters):
    X, T, Xte, Yte = data
    train_size = X.shape[0]
    print('batch_size: %d' % batch_size)
    print('nclusters : %d' % nclusters)
    print('beta      : %d' % beta)
    print('seed      : %d' % seed)
    print('niters    : %d' % niters)
    print('train_size: %d' % train_size)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.set_printoptions(suppress=True)
    np.random.seed(seed)

    model = MLP(dropout=False)
    model.cuda()

    modelg = copy.deepcopy(model)
    gluster = GradientClusterOnline(modelg, beta, nclusters)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    gluster_tc = np.zeros(niters)
    for i in range(niters):
        tic = time.time()
        ids = np.random.permutation(train_size)[:batch_size]
        x = Variable(X[ids[:batch_size]]).cuda()
        t = Variable(T[ids[:batch_size]]).cuda()
        modelg.zero_grad()
        gluster.zero_data()
        y = modelg(x)
        loss = F.nll_loss(y, t)
        loss.backward()
        assign_i = gluster.em_step()
        toc = time.time()
        gluster_tc[i] = (toc-tic)
        # TODO: optim step

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
            b = min(i+batch_size, X.shape)
            yield X[a:b], T[a:b], np.arange(a, b)

    def __len__(self):
        return self.X.shape[0]


def test_gluster_batch(batch_size, data, nclusters, min_size, seed, citers):
    X, T, Xte, Yte = data
    train_size = X.shape[0]
    print('batch_size: %d' % batch_size)
    print('nclusters : %d' % nclusters)
    print('min_size  : %d' % min_size)
    print('seed      : %d' % seed)
    print('citers    : %d' % citers)
    print('train_size: %d' % train_size)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.set_printoptions(suppress=True)
    np.random.seed(seed)

    model = MLP(dropout=False)
    model.cuda()

    modelg = copy.deepcopy(model)
    gluster = GradientClusterBatch(modelg, min_size, nclusters)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    gluster_tc = np.zeros(citers)
    data_loader = DataLoader(X, T, batch_size)
    for i in range(citers):
        tic = time.time()
        gluster.update_batch(data_loader, train_size)
        toc = time.time()
        gluster_tc[i] = (toc-tic)
        # TODO: optim step

    # get test data assignments
    gluster.eval()  # gluster test mode
    x = Variable(Xte).cuda()
    t = Variable(Yte).cuda()
    modelg.zero_grad()
    gluster.zero_data()
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
    print('%.4f' % (toc-tic))
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


def test_mnist(batch_size, epochs, nclusters, min_size, seed, citers, figname):
    print('batch_size: %d' % batch_size)
    print('epochs    : %d' % epochs)
    print('nclusters : %d' % nclusters)
    print('min_size  : %d' % min_size)
    print('seed      : %d' % seed)
    print('citers    : %d' % citers)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.set_printoptions(suppress=True)
    np.random.seed(seed)

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
        shuffle=True)
    # shuffle doesn't matter to gluster as long as dataset is returning index

    model = MLP(dropout=False)
    model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=.01, momentum=0.9,
                          weight_decay=.0005)

    for e in range(epochs):
        train(e, train_loader, model, optimizer)

    modelg = copy.deepcopy(model)
    # model's weight are not going to change, opt.step() is not called
    gluster = GradientClusterBatch(modelg, min_size, nclusters)
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
            assert stat[0] <= total_dist, 'Total distortions went up'
        total_dist, assign_i, target_i, pred_i, loss_i = stat
        toc = time.time()
        gluster_tc[i] = (toc-tic)
        normC = gluster.print_stats()

    print('%.4f +/- %.4f' % (gluster_tc.mean(), gluster_tc.std()))
    torch.save({'assign': assign_i, 'target': target_i,
                'pred': pred_i, 'loss': loss_i,
                'normC': normC},
               figname)
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    # Few iterations
    # data = data_unique_n(100, 5)
    # test_gluster_online(10, data, 5, .9, 1234, 10)

    # More iterations, centers should match input
    # data = data_unique_n(100, 5)
    # test_gluster_online(10, data, 5, .9, 1234, 100)

    # More unique data than centers
    # data = data_unique_n(100, 10)
    # test_gluster_online(10, data, 5, .9, 1234, 100)

    # More centers than data
    # data = data_unique_n(100, 5)
    # test_gluster_online(10, data, 10, .9, 1234, 100)

    # Imbalance data
    # data = data_unique_perc(100, [.9, .1])
    # test_gluster_online(10, data, 2, .9, 1234, 10)

    # More iterations
    # data = data_unique_perc(100, [.9, .1])
    # test_gluster_online(10, data, 2, .9, 1234, 100)

    # Time test
    # data = data_unique_perc(1000, [.9, .1])
    # test_gluster_online(128, data, 2, .9, 1234, 100)
    # 2-3x solwer

    # noise
    # data = data_unique_perc(100, [.9, .1])
    # data = purturb_data(data, .01)
    # test_gluster_online(10, data, 2, .9, 1234, 100)

    # gluster batch
    # data = data_unique_n(100, 5)
    # test_gluster_batch(10, data, 5, 1, 1234, 10)

    # gluster batch noise
    # data = data_unique_perc(100, [.9, .1])
    # data = purturb_data(data, .01)
    # test_gluster_batch(10, data, 2, 1, 12345, 10)

    # MNIST
    citers = 2
    figname = 'notebooks/figs_gluster/mlp,nclusters_2,citers_2.pth.tar'
    test_mnist(128, 2, 2, 10, 1234, citers, figname)

    citers = 10
    figname = 'notebooks/figs_gluster/mlp,nclusters_2,citers_10.pth.tar'
    test_mnist(128, 2, 2, 10, 1234, citers, figname)

    nclusters = 10
    citers = 10
    figname = 'notebooks/figs_gluster/mlp,nclusters_10,citers_10.pth.tar'
    test_mnist(128, 2, nclusters, 10, 1234, citers, figname)
