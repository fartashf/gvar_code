from torch.autograd import Variable
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
from parallel_autograd import ParallelAutograd  # NOQA
from models.mnist import MNISTNet  # NOQA


# autograd on MNIST
batch_size = 128
dim = 100
x = Variable(torch.ones(batch_size, 1, 28, 28), requires_grad=True).cuda()
t = Variable(torch.ones(batch_size).long()).cuda()

model = MNISTNet(dropout=False)
model.cuda()
y = model(x)
loss = F.nll_loss(y, t)
grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
gb = [torch.ones_like(g) for g in grad_params]
grad_x = torch.autograd.grad(grad_params, x, gb)
print(grad_x[0].shape)
# print(grad_x[0].view(batch_size, -1).sum(1))

# # wrong: grad wrt. x
# x = Variable(torch.ones(dim, batch_size), requires_grad=True).cuda()
# W = Variable(torch.ones(1, dim), requires_grad=True).cuda()
# y = W.mm(x)
# loss = y.sum()/batch_size
# grad = torch.autograd.grad(loss, W, create_graph=True)
# grad_x = torch.autograd.grad(grad[0], x, torch.ones_like(grad[0]))
# print(grad_x[0].shape)
# # print(grad_x[0].sum(1, keepdim=True).t())
#
# # wrong: grad wrt. x
# # try twice swap order
# x = Variable(torch.randn(dim, batch_size), requires_grad=True).cuda()
# W = Variable(torch.randn(1, dim), requires_grad=True).cuda()
# y = W.mm(x)
# loss = (y-t.float()).pow(2).sum()/batch_size
# grad = torch.autograd.grad(loss, W, create_graph=True)
# grad_x = torch.autograd.grad(grad[0], x, torch.ones_like(grad[0]))
# print(grad_x[0].shape)
# # print(grad_x[0].sum(0, keepdim=True))
# grad2 = torch.autograd.grad(loss, x, create_graph=True)
# grad_W = torch.autograd.grad(grad2[0], W, grad_x[0])
# print(grad_W[0].shape)
# # print(grad_W[0].sum(0, keepdim=True))

# Jvp
x = Variable(torch.randn(dim, batch_size), requires_grad=True).cuda()
W = Variable(torch.randn(1, dim), requires_grad=True).cuda()
m = torch.ones_like(W)
y = W.mm(x)
loss_ex = (y-t.float()).pow(2).sum(0)/batch_size
loss = loss_ex.sum()

# slow
grads = []
for l in loss_ex:
    grads.append(torch.autograd.grad(l, W, retain_graph=True)[0])
jacob = torch.cat(grads)
jvp0 = jacob.mm(m.view(-1, 1)).view(-1)

# fast
v = Variable(torch.ones_like(loss_ex.data), requires_grad=True).cuda()
grad = torch.autograd.grad(loss_ex, W, v, create_graph=True)
jvp = torch.autograd.grad(grad[0], v, m)[0]

print(jvp.shape)
print((jvp0-jvp).abs().max())

# MNIST
batch_size = 128
x = Variable(torch.randn(batch_size, 1, 28, 28), requires_grad=True).cuda()
t = Variable(torch.ones(batch_size).long()).cuda()

model = MNISTNet(dropout=False)
model.cuda()
y = model(x)
loss_ex = F.nll_loss(y, t, reduce=False)/batch_size
loss = loss_ex.sum()
W = model.parameters()

v = Variable(torch.ones_like(loss_ex.data), requires_grad=True).cuda()
grad_params = torch.autograd.grad(loss_ex, W, v, create_graph=True)
m = [torch.ones_like(g) for g in grad_params]
jvp = torch.autograd.grad(grad_params, v, m)[0]

print(jvp.shape)
# print(jvp.data.cpu().numpy())

import ipdb; ipdb.set_trace()  # NOQA
