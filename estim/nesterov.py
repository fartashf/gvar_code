import torch
import torch.nn
import torch.multiprocessing

from .gestim import GradientEstimator
import models


class NesterovEstimator(GradientEstimator):
    def __init__(self, gamma=0.9, *args, **kwargs):
        super(NesterovEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.gamma = gamma
        self.v = None
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

    def get_new_v(self, model):
        gamma = self.gamma
        params_new = [pp + gamma * vv for pp, vv in
                      zip(model.parameters(), self.v)]
        g = self.grad_at_new_params(model, params_new)

        if self.v is None:
            self.v = [torch.zeros_like(g) for g in model.parameters()]
        v = [gamma * vv + (1-gamma) * gg for vv, gg in zip(self.v, g)]

        return v

    def snap_online(self, model, niters):
        # use g_osnap_iter=1, update vt
        self.v = self.get_new_v(model)

    def grad(self, model, in_place=False):
        v_new = self.get_new_v(model)

        if in_place:
            raise NotImplementedError('not to be used for training')
        return v_new
