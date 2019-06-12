import torch
# import math

from .gestim import GradientEstimator

from cusvd import svdj


class BruteForceFisher(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(BruteForceFisher, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.J = None
        self.batch_size = None

    def grad(self, model_new, in_place=False):
        """
        F = 1/n JJ^T, when J is DxB. F = E[gg^T], where g is Dx1

        If J = U S V^T, JJ^T = U S S^T U^T, then e-values of JJ^T are
        e-values of J, squared.

        We also know that the eigenvalues of cA are just eigenvalues of A
        multiplied by c.
        """
        assert not in_place, 'Not to be used for training.'
        model = model_new

        data = next(self.data_iter)

        J = []
        n = data[0].shape[0]
        for i in range(n):
            loss = model.criterion(model, (data[0][i:i+1], data[1][i:i+1]))
            grad = torch.autograd.grad(loss, model.parameters())
            gf = torch.cat([g.flatten() for g in grad])
            J += [gf]  # /math.sqrt(n)
        self.J = torch.stack(J, dim=1)
        self.batch_size = n

        return grad

    def get_precond_eigs(self):
        S = svdj(self.J, max_sweeps=100)[1]
        # S = torch.svd(self.J)[1]
        return S*S/self.batch_size


class BruteForceFisherFull(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(BruteForceFisherFull, self).__init__(*args, **kwargs)
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
