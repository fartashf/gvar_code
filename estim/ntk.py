import torch
import torch.nn
import torch.multiprocessing

from args import opt_to_ntk_kwargs
from .gestim import GradientEstimator
from ntk.ntk import NeuralTangentKernel


class NTKEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NTKEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.ntk = None

    def grad(self, model_new, in_place=False):
        model = model_new
        if self.ntk is None:
            self.ntk = NeuralTangentKernel(
                model, **opt_to_ntk_kwargs(self.opt))

        data = next(self.data_iter)

        loss = model.criterion(model, data, reduction='none')
        loss0 = loss.mean()

        loss0.backward(retain_graph=True)
        Ki = self.ntk.get_kernel_inverse()

        # optim
        model.zero_grad()
        n = data[0].shape[0]
        loss_ntk = Ki.sum(0).dot(loss)/n

        if in_place:
            loss_ntk.backward()
            return loss0
        g = torch.autograd.grad(loss_ntk, model.parameters())
        return g
