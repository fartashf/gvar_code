import torch
import torch.nn
import torch.multiprocessing

from .gestim import GradientEstimator
import models


class IGTEstimator(GradientEstimator):
    """
    "Reducing the variance in online optimization by transporting past
    gradients"
    https://arxiv.org/pdf/1906.03532.pdf
    """
    def __init__(self, *args, **kwargs):
        super(IGTEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.gamma = 0
        self.theta_tminus1 = []
        self.v = []
        self.model_clone = None

    def grad_at_new_params(self, model, params):
        data = next(self.data_iter)

        # Clone a model to compute gradient at a transported theta
        if self.model_clone is None:
            self.model_clone = models.init_model(self.opt)
            self.model_clone.load_state_dict(model.state_dict())

        for v, w in zip(self.model_clone, params):
            v.copy_(w)
        loss = self.model_clone.criterion(self.model_clone, data)
        g = torch.autograd.grad(loss, self.model_clone.parameters())

        return g

    def snap_online(self, theta_t, niters):
        # use g_osnap_iter=1, update vt
        gamma = self.gamma = niters/(niters+1)
        theta_tminus1 = self.theta_tminus1
        theta_igt = theta_t + gamma/(1-gamma) * (theta_t - theta_tminus1)
        self.v = [gamma * vv + (1-gamma) * ww
                  for vv, ww in zip(self.v, self.grad(theta_igt))]
        self.theta_tminus1 = theta_t

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        # Clone a model to compute gradient at a transported theta
        if self.model_clone is None:
            self.model_clone = models.init_model(self.opt)
            self.model_clone.load_state_dict(model.state_dict())

        t = self.niters
        theta_t = list(model.parameters())
        theta_tminus1 = self.theta_tminus1
        theta_igt = list(self.model_clone.parameters())

        gamma = t/(t+1)
        for th_igt, th_t, th_tm1 in zip(theta_igt, theta_t, theta_tminus1):
            th_igt.copy_(th_t + gamma/(1-gamma) * (th_t - th_tm1))

        loss = self.model_clone.criterion(self.model_clone, data)
        g_igt = torch.autograd.grad(loss, theta_igt)
        v = [gamma * vtm1 + (1-gamma) * igt
             for vtm1, igt in zip(self.v, g_igt)]
        wt = self.mu * self.wtm1 - self.alpha * v

        # TODO: need to get theta_t-1 after heavy ball update but this needs
        # to be done in the optimizer.
        self.theta_tminus1 = [w.clone() for w in wt]

        if in_place:
            return loss
        return g
