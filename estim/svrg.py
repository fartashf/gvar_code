import logging
import copy

import torch
import torch.nn
import torch.multiprocessing

from log_utils import Profiler
from data import InfiniteLoader
from .gestim import GradientEstimator


class SVRGEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SVRGEstimator, self).__init__(*args, **kwargs)
        self.data_iter = iter(InfiniteLoader(self.data_loader))
        self.mu = []

    def snap_batch(self, model, niters):
        self.model = model = copy.deepcopy(model)
        self.mu = [torch.zeros_like(g) for g in model.parameters()]
        num = 0
        batch_time = Profiler()
        for batch_idx, data in enumerate(self.data_loader):
            idx = data[2]
            num += len(idx)
            loss = model.criterion(model, data, reduction='sum')
            grad_params = torch.autograd.grad(loss, model.parameters())
            for m, g in zip(self.mu, grad_params):
                m += g
            batch_time.toc('Time')
            batch_time.end()
            if batch_idx % 10 == 0:
                logging.info(
                        'SVRG Snap> [{0}/{1}]: {bt}'.format(
                            batch_idx, len(self.data_loader),
                            bt=str(batch_time)))
        for m in self.mu:
            m /= num

    def grad(self, model_new, in_place=False):
        data = next(self.data_iter)

        model_old = self.model

        # old grad
        loss = model_old.criterion(model_old, data)
        g_old = torch.autograd.grad(loss, model_old.parameters())

        # new grad
        loss = model_new.criterion(model_new, data)
        g_new = torch.autograd.grad(loss, model_new.parameters())

        if in_place:
            for m, go, gn, p in zip(
                    self.mu, g_old, g_new, model_new.parameters()):
                p.grad.copy_(m-go+gn)
            return loss
        ge = [m-go+gn for m, go, gn in zip(self.mu, g_old, g_new)]
        return ge

    def state_dict(self):
        return {'svrg.mu': [m.cpu() for m in self.mu]}

    def load_state_dict(self, state):
        if 'svrg.mu' not in state:
            return
        mu = state['svrg.mu']
        for mx, my in zip(mu, self.mu):
            mx.copy_(my)
