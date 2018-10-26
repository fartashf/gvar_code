import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import sys
import time
sys.path.append('../')
from models.mnist import MLP, MNISTNet  # NOQA
from gluster import GradientCluster  # NOQA


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
    normC = np.sum(L, 0)
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


def test_gluster(batch_size, data, nclusters, beta, seed, niters):
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
    gluster = GradientCluster(modelg, nclusters, beta)
    # Test if Gluster can be disabled
    # gluster.deactivate()

    gluster_tc = np.zeros(niters)
    for i in range(niters):
        tic = time.time()
        ids = np.random.permutation(train_size)[:batch_size]
        x = Variable(X[ids[:batch_size]]).cuda()
        t = Variable(T[ids[:batch_size]]).cuda()
        modelg.zero_grad()
        gluster.zero()
        y = modelg(x)
        loss = F.nll_loss(y, t)
        loss.backward()
        assign_i = gluster.em_update()
        toc = time.time()
        gluster_tc[i] = (toc-tic)
        # TODO: optim step

    print('assign:')
    print(assign_i)
    # use model to prevent C update
    print_stats(model, gluster, Xte, Yte, batch_size)

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

    print('Time:')
    print('%.4f +/- %.4f' % (gluster_tc.mean(), gluster_tc.std()))
    print('%.4f +/- %.4f' % (model_tc.mean(), model_tc.std()))
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    # Few iterations
    # data = data_unique_n(100, 5)
    # test_gluster(10, data, 5, .9, 1234, 10)

    # More iterations, centers should match input
    # data = data_unique_n(100, 5)
    # test_gluster(10, data, 5, .9, 1234, 100)

    # More unique data than centers
    # data = data_unique_n(100, 10)
    # test_gluster(10, data, 5, .9, 1234, 100)

    # More centers than data
    # data = data_unique_n(100, 5)
    # test_gluster(10, data, 10, .9, 1234, 100)

    # Imbalance data
    # data = data_unique_perc(100, [.9, .1])
    # test_gluster(10, data, 2, .9, 1234, 10)

    # More iterations
    # data = data_unique_perc(100, [.9, .1])
    # test_gluster(10, data, 2, .9, 1234, 100)

    # Time test
    # data = data_unique_perc(1000, [.9, .1])
    # test_gluster(128, data, 2, .9, 1234, 100)
    # 2-3x solwer

    # noise
    data = data_unique_perc(100, [.9, .1])
    data = purturb_data(data, .01)
    test_gluster(10, data, 2, .9, 1234, 100)
