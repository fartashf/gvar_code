from torch.autograd import Variable
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
from parallel_autograd import ParallelAutograd  # NOQA
from models.mnist import MNISTNet  # NOQA


x = Variable(torch.ones(128, 1, 28, 28)).cuda()
t = Variable(torch.ones(128).long()).cuda()

model = MNISTNet()
model.cuda()
y = model(x)
loss = F.nll_loss(y, t)
model.zero_grad()
loss.backward()

model = ParallelAutograd(MNISTNet())
model.cuda()
y = model(x)
# loss = F.nll_loss(y, t, reduce=False)
# model.zero_grad()
# loss.backward()

import ipdb; ipdb.set_trace()  # NOQA
