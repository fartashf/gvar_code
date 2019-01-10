import torch
import torch.nn
import torch.nn.functional as F
import torch.multiprocessing

from data import InfiniteLoader
from .gestim import GradientEstimator


class SGDEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SGDEstimator, self).__init__(*args, **kwargs)
        # many open files? torch.multiprocessing sharing file_system
        self.data_iter = iter(InfiniteLoader(self.data_loader))

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # print(loss)
        # import ipdb; ipdb.set_trace()
        if in_place:
            loss.backward()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        return g
