import torch
import torch.nn
import torch.multiprocessing

from .gestim import GradientEstimator


class HeavyBallEstimator(GradientEstimator):
    """
    Heavy-ball momentum
    """
    def __init__(self, gamma=0.9, *args, **kwargs):
        super(HeavyBallEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.gamma = gamma
        self.v = None

    def get_new_v(self, model):
        data = next(self.data_iter)

        loss = model.criterion(model, data)
        g = torch.autograd.grad(loss, model.parameters())

        gamma = self.gamma
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
