import torch

from .gestim import GradientEstimator

from cusvd import svdj
from hf import lanczos
from hf.utils import fisher_vec_bk, fisher_vec_fw


class LanczosEstimator(GradientEstimator):
    def __init__(self, *args, lanczos_method='fw', **kwargs):
        super(LanczosEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.Q = None
        self.method = lanczos_method

    def grad(self, model, in_place=False):
        assert not in_place, 'Not to be used for training.'
        data = next(self.data_iter)
        model.double()
        data[0] = data[0].double()
        loss = model.criterion(model, data, reduction='none')
        if self.method == 'fw':
            fisher_vec = fisher_vec_fw
        else:
            loss = loss.mean()
            fisher_vec = fisher_vec_bk
        params = list(model.parameters())

        batch_size = data[0].shape[0]
        self.Q = lanczos.lanczos_torch(loss, params,
                                       num_iter=batch_size+1,
                                       EPS=1e-15,
                                       dot_op=fisher_vec)[0].float()
        # assert self.Q.shape[1] < batch_size+1, 'Too many vectors'

        grad = torch.autograd.grad(loss.mean(), params)
        model.float()

        return [g.float() for g in grad]

    def get_precond_eigs(self):
        S = svdj(self.Q, max_sweeps=100)[1]
        return S
