import torch
import sys
sys.path.append('../')
from models.mnist import MNISTNet  # NOQA


def f1(*args, **kwargs):
    print('f1')


def f2(*args, **kwargs):
    print('f2')


x = torch.ones(128, 1, 28, 28).cuda()
t = torch.ones(128).long().cuda()

model = MNISTNet()
model.cuda()

model.conv1.register_forward_pre_hook(f1)
model.conv1.register_forward_pre_hook(f2)

y = model(x)
