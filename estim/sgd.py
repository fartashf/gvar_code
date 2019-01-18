import torch
import torch.nn
import torch.multiprocessing

from data import InfiniteLoader
from .gestim import GradientEstimator


class SGDEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SGDEstimator, self).__init__(*args, **kwargs)
        # many open files? torch.multiprocessing sharing file_system
        self.init_data_iter()

    def init_data_iter(self):
        self.data_iter = iter(InfiniteLoader(self.data_loader))
        self.estim_iter = iter(InfiniteLoader(self.data_loader))

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        loss = model.criterion(model, data)

        # print(loss)
        # import ipdb; ipdb.set_trace()
        if in_place:
            loss.backward()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        return g

    def grad_estim(self, model):
        # insuring continuity of data seen in training
        # TODO: make sure sub-classes never use any other data_iter, e.g. raw
        dt = self.data_iter
        self.data_iter = self.estim_iter
        ret = self.grad(model)
        self.data_iter = dt
        return ret
