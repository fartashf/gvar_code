from torch.autograd import Variable
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
from parallel_autograd import ParallelAutograd  # NOQA
from models.mnist import MNISTNet  # NOQA


batch_size = 128
x = Variable(torch.ones(batch_size, 1, 28, 28)).cuda()
t = Variable(torch.ones(batch_size).long()).cuda()

model = MNISTNet(dropout=False)
model.cuda()
y = model(x)
loss = F.nll_loss(y, t)
model.zero_grad()
loss.backward()

model_p = ParallelAutograd(model, batch_size)
model_p.cuda()
y_p = model_p(x)
# y_p = model_p(x)  # check if dropout is disabled
print(torch.pow(y-y_p, 2).mean())
loss = F.nll_loss(y_p, t)
model_p.zero_grad()
loss.backward()
params = list(model_p.parameters())

import ipdb; ipdb.set_trace()  # NOQA
