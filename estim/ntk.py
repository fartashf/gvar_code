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
        # loss0 = loss.mean()
        loss0 = loss.sum()
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
            return loss0/n
        g = torch.autograd.grad(loss_ntk, model.parameters())
        return g

    def get_precond_eigs_nodata(self):
        return self.S


class NeuralTangentKernelFull(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NeuralTangentKernelFull, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.K = None
        self.damping = self.opt.ntk_damping

    def grad(self, model_new, in_place=False, data=None):
        # assert not in_place, 'Not to be used for training.'

        model = model_new

        if data is None:
            data = next(self.data_iter)

        K = 0
        n = data[0].shape[0]
        J = []
        loss0 = 0
        for i in range(n):
            loss = model.criterion(model, (data[0][i:i+1], data[1][i:i+1]))
            with torch.no_grad():
                loss0 += loss
            grad = torch.autograd.grad(loss, model.parameters())
            J += [torch.cat([g.flatten() for g in grad])]
        J = torch.stack(J)
        K = J @ J.t()
        K /= n
        self.K = K.clone()

        # U, S, V = svdj(self.J, max_sweeps=100)
        ftype = 2
        if ftype == 1:
            # using svd of J
            U, S, V = torch.svd(J.t())
            Si = 1./(S*S/n+self.damping)
            grad = ((V.t() @ Si.diag() @ V) @ J).sum(0)
        else:
            # using svd of K
            U, S, V = torch.svd(J @ J.t())
            Si = 1./(S/n+self.damping)
            Ki = U @ Si.diag() @ V.t()
            grad = Ki.sum(0) @ J
        if in_place:
            curi = 0
            for p in model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(grad[curi:curi+p.numel()].view(p.shape))
                curi += p.numel()
            return loss0/n
        return grad

    def get_precond_eigs_nodata(self):
        S = svdj(self.K, max_sweeps=100)[1]
        return S
