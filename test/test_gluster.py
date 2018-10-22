import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import sys
sys.path.append('../')
from models.mnist import MLP, MNISTNet  # NOQA
from gluster import GradientCluster  # NOQA


def print_stats(model, gluster, X, T, batch_size):
    nunique = X.shape[0]
    W = list([param for name, param in model.named_parameters()])
    print(list([name for name, param in model.named_parameters()]))

    centers = []
    for param, value in gluster.centers.iteritems():
        if param.dim() == 1:
            Cb = value
            centers += [Cb]
        elif param.dim() == 2:
            (Ci, Co) = value
            Cf = torch.matmul(Co.unsqueeze(-1), Ci.unsqueeze(1))
            centers += [Cf]
            assert Cf.shape == (
                Ci.shape[0], Co.shape[1], Ci.shape[1]), 'Cf: C x d_out x d_in'
    nclusters = centers[0].shape[0]

    dist = np.zeros((nunique, nclusters))
    normG = np.zeros((nunique,))
    normC = np.zeros((nclusters,))
    for i in range(nunique):
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
         for g in centers]
    normC = np.sum(L, 0)
    print('Dist:')
    print(dist)
    print('normG:')
    print(normG)
    print('normC:')
    print(normC)
    print('Reinit count: %s' % str(gluster.reinits))
    print('Cluster size: %s' % str(gluster.cluster_size))


def test_gluster(batch_size, train_size, nclusters, nunique, beta, seed,
                 niters):
    print('batch_size: %d' % batch_size)
    print('train_size: %d' % train_size)
    print('nclusters : %d' % nclusters)
    print('nunique   : %d' % nunique)
    print('beta      : %d' % beta)
    print('seed      : %d' % seed)
    print('niters    : %d' % niters)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.set_printoptions(suppress=True)
    np.random.seed(seed)

    X = torch.rand(train_size, 1, 28, 28)
    T = torch.ones(train_size).long()

    for i in range(nunique, train_size, nunique):
        X[i:i+nunique].copy_(X[:nunique])

    model = MLP(dropout=False)
    model.cuda()

    modelg = copy.deepcopy(model)
    gluster = GradientCluster(modelg, nclusters, beta)

    for i in range(niters):
        ids = np.random.permutation(train_size)[:batch_size]
        x = Variable(X[ids[:batch_size]]).cuda()
        t = Variable(T[ids[:batch_size]]).cuda()
        modelg.zero_grad()
        gluster.zero()
        y = modelg(x)
        loss = F.nll_loss(y, t)
        loss.backward()
        assign_i = gluster.em_update()
        # TODO: optim step

    print('assign:')
    print(assign_i)
    # use model to prevent C update
    print_stats(model, gluster,
                X[:2*nunique], T[:2*nunique],
                batch_size)
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_gluster(10, 100, 5, 5, .9, 1234, 10)
    # test_gluster(10, 100, 5, 5, .9, 1234, 100)
    # test_gluster(10, 100, 5, 10, .9, 1234, 100)
    # test_gluster(10, 100, 10, 5, .9, 1234, 100)
