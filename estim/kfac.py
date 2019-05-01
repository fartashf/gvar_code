import torch
import torch.nn
import torch.multiprocessing

from .gestim import GradientEstimator


class KFACEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(KFACEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.optim = None  # this is set after optim is initialized

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        loss, outputs = model.criterion(model, data, return_output=True)
        if self.optim.steps % self.optim.TCov == 0:
            self.snap_TCov(model, loss, outputs)

        if in_place:
            loss.backward()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        return g
