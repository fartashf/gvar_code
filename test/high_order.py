import torch
from torchvision.models import resnet18
from torch.autograd import Variable

model = resnet18().cuda()

# dummy inputs for the example
input = Variable(torch.randn(2, 3, 224, 224).cuda(), requires_grad=True)
target = Variable(torch.zeros(2).long().cuda())

# as usual
output = model(input)
loss = torch.nn.functional.nll_loss(output, target)

grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
# torch.autograd.grad does not accumuate the gradients into the .grad
# attributes
# It instead returns the gradients as Variable tuples.

# now compute the 2-norm of the grad_params
grad_norm = 0
for grad in grad_params:
    grad_norm += grad.pow(2).sum()
grad_norm = grad_norm.sqrt()

# take the gradients wrt grad_norm. backward() will accumulate
# the gradients into the .grad attributes
grad_norm.backward()

import ipdb; ipdb.set_trace()  # NOQA
