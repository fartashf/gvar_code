import torch
import torch.nn
import torch.multiprocessing

from args import opt_to_ntk_kwargs
from .gestim import GradientEstimator
from ntk.ntk import NeuralTangentKernel

from cusvd import svdj


class NeuralTangentKernelEstimator(GradientEstimator):
    def __init__(self, empirical=False, *args, **kwargs):
        super(NeuralTangentKernelEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.ntk = None
        self.Ki = None
        self.S = None
        self.divn = self.opt.ntk_divn
        self.empirical = empirical

    def grad(self, model_new, in_place=False, data=None):
        model = model_new
        if self.ntk is None:
            self.ntk = NeuralTangentKernel(
                model, **opt_to_ntk_kwargs(self.opt))

        if data is None:
            data = next(self.data_iter)

        loss0, Ki = self.snap_K(model, data)
        self.Ki = Ki

        # optim
        model.zero_grad()
        n = data[0].shape[0]
        loss_ntk = Ki.sum(0) @ loss0/n

        if in_place:
            loss_ntk.backward()
            return loss0.mean()
        g = torch.autograd.grad(loss_ntk, model.parameters())
        return g

    def snap_K(self, model, data):
        model.zero_grad()
        self.ntk.activate()
        loss0, output = model.criterion(model, data, reduction='none',
                                        return_output=True)
        target = data[1].cuda()
        loss_sample = model.criterion.loss_sample(
            output, target, self.empirical).sum()
        loss_sample.backward(retain_graph=True)

        Ki, self.S = self.ntk.get_kernel_inverse()
        self.ntk.deactivate()
        model.zero_grad()
        return loss0, Ki

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

        ftype = 4
        if ftype == 1:
            # using svd of J
            U, S, V = torch.svd(J.t())
            Si = 1./(S*S/n+self.damping)
            grad = ((V.t() @ Si.diag() @ V) @ (J/n)).sum(0)
        elif ftype == 2:
            # using svd of J J'
            U, S, V = torch.svd(J @ J.t())
            Si = 1./(S/n+self.damping)
            Ki = U @ Si.diag() @ V.t()
            grad = Ki.sum(0) @ J/n
        elif ftype == 3:
            # using svd of K
            U, S, V = torch.svd(K)
            Si = 1./(S+self.damping)
            Ki = U @ Si.diag() @ V.t()
            grad = Ki.sum(0) @ J/n
        else:
            # using svdj
            U, S, V = svdj(K, max_sweeps=100)
            Si = 1./(S+self.damping)
            Ki = U @ Si.diag() @ V.t()
            grad = Ki.sum(0) @ J/n
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
