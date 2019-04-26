import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import ntk.layer
import logging
from log_utils import Profiler


class NeuralTangentKernel(object):
    def __init__(self, model, debug=True, **kwargs):
        self.model = model
        self.debug = debug
        self.ntk = ntk.layer.Container(model, debug=debug, **kwargs)

    def activate(self):
        self.G.activate()

    def eval(self):
        self.G.eval()

    def deactivate(self):
        self.G.deactivate()

    def copy_(self, model):
        for m, g in zip(model.parameters(), self.model.parameters()):
            g.data.copy_(m.data)

    def get_dot(self):
        batch_dist = self.G.get_dist()
        total_dist = torch.stack(batch_dist).sum(0)
        return GoG
