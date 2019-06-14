import torch

from .gestim import GradientEstimator

from cusvd import svdj
from hf import lanczos
from hf.utils import fisher_vec_bk, fisher_vec_fw


class LanczosEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(LanczosEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.method = self.opt.lanczos_method
        self.T = None
        self.batch_size = None

    def grad(self, model, in_place=False, data=None):
        """
        The empirical fisher is 1/n sum_i out_prod(g_i, g_i).

        fisher_vec_fw can compute the Jv for multi-dimensional output, then
        computes JJv as sum_o J_o J_o v, for every output dimension o.
        dividing this by n afterwards gives us empirical fisher.

        fisher_vec_bk can compute Jv for scalar output, which means Jv
        computed using bk is 1xD dimensional, then JJv computed using this
        method is equal to J*vec{1}*vec{1}^T * J v, where J is the jacobian of
        the multi-dimensional output before sum.
        vec{1}*vec{1}^T=matrix{1}_{dxd}, which is different from the
        above computation where we have Identity_{dxd}.
        """
        assert not in_place, 'Not to be used for training.'
        if data is None:
            data = next(self.data_iter)
        model.double()
        data[0] = data[0].double()
        loss = model.criterion(model, data, reduction='none')
        batch_size = data[0].shape[0]
        if self.method == 'fw':
            # loss = loss.mean()
            fisher_vec = fisher_vec_fw
            # divn = batch_size
        else:
            loss = loss.sum()
            # loss = loss.mean()
            fisher_vec = fisher_vec_bk
            # divn = batch_size
        params = list(model.parameters())

        Q, beta, gamma = lanczos.lanczos_torch(loss, params,
                                               num_iter=batch_size+1,
                                               EPS=1e-15,
                                               dot_op=fisher_vec,
                                               divn=1)
        beta, gamma = torch.Tensor(beta).cuda(), torch.Tensor(gamma).cuda()
        self.T = lanczos.tridiag(gamma, beta, beta)
        self.batch_size = batch_size
        # assert self.Q.shape[1] < batch_size+1, 'Too many vectors'

        grad = torch.autograd.grad(loss.sum()/batch_size, params)
        # grad = torch.autograd.grad(loss, params)
        model.float()
        data[0] = data[0].float()

        return [g.float() for g in grad]

    def get_precond_eigs_nodata(self):
        S = svdj(self.T, max_sweeps=100)[1]
        # S = torch.svd(self.T)[1]
        return S/self.batch_size
