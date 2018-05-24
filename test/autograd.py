import torch
import torch.autograd
from torch.autograd import Variable
import nonacc_autograd


x = Variable(torch.randn(10, 2))
W = Variable(torch.randn(1, 10), requires_grad=True)
y = torch.mm(W, x)
grad = torch.randn(1, 2)
torch.autograd.backward([y], [grad], retain_graph=True)
print(W.grad.shape)
print(W.grad)
torch.autograd.backward([y[0, 0:1], y[0, 1:2]],
                        [grad[0, 0:1], grad[0, 1:2]],
                        retain_graph=True)
print(W.grad.shape)
print(W.grad)
wgrad = torch.autograd.grad([y[0, 0:1], y[0, 1:2]],
                            [W],
                            [grad[0, 0:1], grad[0, 1:2]],
                            retain_graph=True)
print(len(wgrad))
print(wgrad)
# https://discuss.pytorch.org/t/quickly-get-individual-gradients-not-sum-of-gradients-of-all-network-outputs/8405
# autograd.grad does not support this
wgrad = nonacc_autograd.grad([y[0, 0:1], y[0, 1:2]],
                             [W],
                             [grad[0, 0:1], grad[0, 1:2]],
                             retain_graph=True)
print(len(wgrad))
print(wgrad)
import ipdb; ipdb.set_trace()
