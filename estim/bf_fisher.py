import torch
import torch.nn
import torch.multiprocessing

from .gestim import GradientEstimator

from cusvd import svdj


class BruteForceFisher(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(BruteForceFisher, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.F = None

    def grad(self, model_new, in_place=False):
        assert not in_place, 'Not to be used for training.'
        model = model_new

        data = next(self.data_iter)

        F = 0
        n = data[0].shape[0]
        for i in range(n):
            loss = model.criterion(model, (data[0][i:i+1], data[1][i:i+1]))
            grad = torch.autograd.grad(loss, model.parameters())
            gf = torch.cat([g.flatten() for g in grad])
            F += torch.einsum('i,j->ij', gf, gf)
        F /= n
        self.F = F.clone()

        return grad

    def get_precond_eigs(self):
        S = svdj(self.F, max_sweeps=100)[1]
        return S
