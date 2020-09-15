import torch
import torch.nn
import torch.multiprocessing

from .gestim import GradientEstimator


class AdamEstimator(GradientEstimator):
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, *args, **kwargs):
        super(AdamEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.beta1, self.beta2 = beta1, beta2
        self.v = None
        self.m = None
        self.beta1_correction, self.beta2_correction = 1, 1
        self.eps = eps

    def get_new_mv(self, model):
        data = next(self.data_iter)
        loss = model.criterion(model, data)
        g = torch.autograd.grad(loss, model.parameters())

        beta1, beta2 = self.beta1, self.beta2
        if self.v is None:
            self.m = [torch.zeros_like(g) for g in model.parameters()]
            self.v = [torch.zeros_like(g) for g in model.parameters()]
        m = [beta1 * mm + (1-beta1) * gg for mm, gg in zip(self.m, g)]
        v = [beta1 * vv + (1-beta2) * gg for vv, gg in zip(self.v, g)]

        return m, v

    def snap_online(self, model, niters):
        # use g_osnap_iter=1, update vt
        self.m, self.v = self.get_new_mv(model)
        self.beta1_correction = 1 - self.beta1 ** niters
        self.beta2_correction = 1 - self.beta2 ** niters

    def grad(self, model, in_place=False):
        m, v = self.get_new_mv(model)
        mh = [mm/self.beta1_correction for mm in m]
        vh = [vv/self.beta2_correction for vv in v]
        g = [mm/(vv.sqrt()+self.eps) for mm, vv in zip(mh, vh)]

        if in_place:
            raise NotImplementedError('not to be used for training')
        return g
