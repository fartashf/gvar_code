import torch
import torch.nn
import torch.multiprocessing

from args import opt_to_ntk_kwargs
from .gestim import GradientEstimator
from ntk.ntk import NeuralTangentKernel

from cusvd import svdj


class NeuralTangentKernelEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NeuralTangentKernelEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.ntk = None
        self.Ki = None
        self.S = None

    def grad(self, model_new, in_place=False, data=None):
        model = model_new
        if self.ntk is None:
            self.ntk = NeuralTangentKernel(
                model, **opt_to_ntk_kwargs(self.opt))

        if data is None:
            data = next(self.data_iter)

        self.ntk.activate()
        loss = model.criterion(model, data, reduction='none')
        loss0 = loss.mean()

        loss0.backward(retain_graph=True)
        Ki, self.S = self.ntk.get_kernel_inverse()
        self.Ki = Ki
        self.ntk.deactivate()

        # optim
        model.zero_grad()
        n = data[0].shape[0]
        loss_ntk = Ki.sum(0).dot(loss)/n

        if in_place:
            loss_ntk.backward()
            return loss0
        g = torch.autograd.grad(loss_ntk, model.parameters())
        return g

    def get_precond_eigs_nodata(self):
        return self.S


class NeuralTangentKernelFull(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NeuralTangentKernelFull, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.K = None

    def grad(self, model_new, in_place=False, data=None):
        assert not in_place, 'Not to be used for training.'
        model = model_new

        if data is None:
            data = next(self.data_iter)

        K = 0
        n = data[0].shape[0]
        J = []
        for i in range(n):
            loss = model.criterion(model, (data[0][i:i+1], data[1][i:i+1]))
            grad = torch.autograd.grad(loss, model.parameters())
            J += [torch.cat([g.flatten() for g in grad])]
        J = torch.stack(J)
        K = J @ J.t()
        K /= n
        self.K = K.clone()

        return grad

    def get_precond_eigs_nodata(self):
        S = svdj(self.K, max_sweeps=100)[1]
        return S
