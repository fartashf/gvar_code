import torch
import math

from .gestim import GradientEstimator

from cusvd import svdj


class BruteForceFisher(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(BruteForceFisher, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.J = None
        self.batch_size = None
        self.damping = self.opt.kf_damping
        self.sqrt = self.opt.kf_sqrt

    def grad(self, model_new, in_place=False, data=None):
        """
        F = 1/n JJ^T, when J is DxB. F = E[gg^T], where g is Dx1

        If J = U S V^T, JJ^T = U S S^T U^T, then e-values of JJ^T are
        e-values of J, squared.

        We also know that the eigenvalues of cA are just eigenvalues of A
        multiplied by c.
        """
        # assert not in_place, 'Not to be used for training.'
        model = model_new

        if data is None:
            data = next(self.data_iter)

        J = []
        n = data[0].shape[0]
        loss0 = 0
        for i in range(n):
            loss = model.criterion(model, (data[0][i:i+1], data[1][i:i+1]))/n
            with torch.no_grad():
                loss0 += loss
            grad = torch.autograd.grad(loss, model.parameters())
            gf = torch.cat([g.flatten() for g in grad])
            J += [gf]
        self.J = torch.stack(J, dim=1)
        self.batch_size = n

        g = self.J.sum(1)
        U, S, V = svdj(self.J, max_sweeps=100)
        # U, S, V = torch.svd(self.J)
        # eps = 1e-10
        # S.mul_((S > eps).float())
        if self.sqrt:
            Si = S*math.sqrt(n)
        else:
            Si = S*S*n
        Si += self.damping
        grad = U @ ((U.t() @ g) / Si)
        if in_place:
            curi = 0
            for p in model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.copy_(grad[curi:curi+p.numel()].view(p.shape))
                    curi += p.numel()
            return loss0
        return grad

    def get_precond_eigs_nodata(self):
        S = svdj(self.J, max_sweeps=100)[1]
        # S = torch.svd(self.J)[1]
        return S*S*self.batch_size


class BruteForceFisherFull(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(BruteForceFisherFull, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.F = None

    def grad(self, model_new, in_place=False, data=None):
        assert not in_place, 'Not to be used for training.'
        model = model_new

        if data is None:
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

    def get_precond_eigs_nodata(self):
        S = svdj(self.F, max_sweeps=100)[1]
        return S
