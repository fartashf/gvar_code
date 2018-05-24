import torch
import torch.nn
import numpy as np
import torchvision.models
import sys
sys.path.append('../')
import models  # NOQA
import models.cifar10  # NOQA


model = torch.nn.DataParallel(
    models.cifar10.__dict__['resnet32']())
model.cuda()
A = list(model.parameters())
B = [a.numel() for a in A]
print(np.sum(B)*4./1024./1024./1024)
model = torchvision.models.vgg19()
A = list(model.parameters())
B = [a.numel() for a in A]
print(np.sum(B)*4./1024./1024./1024)
import ipdb; ipdb.set_trace()
