import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class Model(nn.Module):
    def __init__(self, arch, pretrained=False):
        super(Model, self).__init__()
        model = torchvision.models.__dict__[arch](pretrained)
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=-1)
