import logging
import copy

import torch
import torch.nn
import torch.nn.functional as F
import torch.multiprocessing

from log_utils import Profiler
from data import InfiniteLoader
from .gestim import GradientEstimator


class SVRGEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SVRGEstimator, self).__init__(*args, **kwargs)
        self.data_iter = iter(InfiniteLoader(self.data_loader))

    def snap_batch(self, model, niters):
        self.model = model = copy.deepcopy(model)
        self.mu = [torch.zeros_like(g) for g in model.parameters()]
        num = 0
        batch_time = Profiler()
        for batch_idx, (data, target, idx) in enumerate(self.data_loader):
            num += len(idx)
            data, target = data.cuda(), target.cuda()
            model.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
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
        data, target = data[0].cuda(), data[1].cuda()

        # old grad
        model_old.zero_grad()
        output = model_old(data)
        loss = F.nll_loss(output, target)
        g_old = torch.autograd.grad(loss, model_old.parameters())

        # new grad
        model_new.zero_grad()
        output = model_new(data)
        loss = F.nll_loss(output, target)
        g_new = torch.autograd.grad(loss, model_new.parameters())

        if in_place:
            for m, go, gn, p in zip(
                    self.mu, g_old, g_new, model_new.parameters()):
                p.grad.copy_(m-go+gn)
            return loss
        ge = [m-go+gn for m, go, gn in zip(self.mu, g_old, g_new)]
        return ge
