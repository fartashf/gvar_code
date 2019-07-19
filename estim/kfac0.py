import torch
import torch.nn
import torch.multiprocessing

from args import opt_to_kfac_kwargs
from .gestim import GradientEstimator
import kfac.layer


class KFACZeroEstimator(GradientEstimator):
    def __init__(self, g_estim='kfac0', empirical=False, *args, **kwargs):
        super(KFACZeroEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.kfac = None
        self.empirical = empirical
        self.g_estim = g_estim

    def grad(self, model_new, in_place=False, data=None):
        model = model_new
        if self.kfac is None:
            self.kfac = kfac.layer.Container(
                model, **opt_to_kfac_kwargs(self.opt, self.g_estim))

        if data is None:
            data = next(self.data_iter)

        loss = self.snap_TCov(model, data)
        self.kfac.update_inv()
        # data = next(self.data_iter)
        # loss = model.criterion(model, data)

        if in_place:
            loss.backward()
            self.kfac.precond_inplace()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        # g = self.kfac.precond_outplace(g)
        return g

    def get_precond_eigs_nodata(self):
        return torch.cat(self.kfac.get_precond_eigs())

    def snap_TCov(self, model, data):
        self.kfac.activate()
        model.zero_grad()
        loss, output = model.criterion(model, data, return_output=True)
        target = data[1].cuda()
        loss_sample = model.criterion.loss_sample(
            output, target, self.empirical).mean()
        loss_sample.backward(retain_graph=True)
        model.zero_grad()  # clear the gradient for computing true-fisher.
        self.kfac.deactivate()
        return loss
