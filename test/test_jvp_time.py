from torch.autograd import Variable
import torch
import torch.nn.functional as F
# import torchvision.models as models
import time
import sys
sys.path.append('../')
import models.cifar10  as models  # NOQA


def dbp(model):
    retain_graph = False
    W = model.parameters()
    batch_size = 128
    # batch_size = 8
    # x = Variable(torch.randn(batch_size, 3, 224, 224)).cuda()
    x = Variable(torch.randn(batch_size, 3, 32, 32)).cuda()
    t = Variable(torch.ones(batch_size).long()).cuda()

    y = model(x)
    tic = time.time()
    loss_ex = F.nll_loss(y, t, reduce=False)/batch_size
    loss = loss_ex.sum()
    t0 = time.time()-tic
    # tic = time.time()
    # loss.backward(retain_graph=True)
    # t1 = time.time()-tic

    tic = time.time()
    v = Variable(torch.ones_like(loss_ex.data), requires_grad=True).cuda()
    grad_params = torch.autograd.grad(loss_ex, W, v, create_graph=True)
    t1 = time.time()-tic
    tic = time.time()
    m = [torch.ones_like(g) for g in grad_params]
    # 1.5s in v0.4.0 vs 3s in v0.3.1
    jvp = torch.autograd.grad(grad_params, v, m,
                              retain_graph=retain_graph)[0]
    t2 = time.time()-tic
    tic = time.time()
    # no need for retain_graph=True in pytorch v0.4.0
    loss.backward(jvp)
    t3 = time.time()-tic
    # del loss, jvp
    return t0, t1, t2, t3


def test_dbp(arch):
    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    num = 2
    model = models.__dict__[arch]()
    model.cuda()
    for i in range(num):
        ttt = dbp(model)
        t0 += ttt[0]
        t1 += ttt[1]
        t2 += ttt[2]
        t3 += ttt[3]
    torch.cuda.empty_cache()

    # print(jvp.shape)
    print('%s forward: %.4fs' % (arch, t0/num))
    print('%s grad 1: %.4fs' % (arch, t1/num))
    print('%s grad 2: %.4fs' % (arch, t2/num))
    print('%s backward: %.4fs' % (arch, t3/num))

    # import ipdb; ipdb.set_trace()  # NOQA


test_dbp('resnet32')
# test_dbp('vgg19')
# test_dbp('resnet34')
# test_dbp('vgg19_bn')
