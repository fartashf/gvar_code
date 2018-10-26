import torch
import sys
from collections import OrderedDict
sys.path.append('../')
from models.mnist import MLP, MNISTNet  # NOQA


class GradientCluster(object):
    def __init__(self, model, nclusters, beta):
        # Q: duplicates
        # TODO: challenge: how many C? memory? time?
        self.layer_cluster = {'Linear': self._linear_cluster,
                              'Conv2d': self._conv2d_cluster}
        self.known_modules = self.layer_cluster.keys()

        self.is_active = True
        self.is_eval = False
        self.nclusters = nclusters
        self.cluster_size = torch.ones(nclusters, 1).cuda()
        self.reinits = torch.ones(nclusters, 1).long().cuda()
        self.beta = beta  # cluster size decay factor
        # self.train_size = train_size
        self.model = model
        self.modules = []
        self.zero()
        self._init_centers()
        # self.assignments = torch.zeros(train_size).long().cuda()

        self._register_hooks()

    def activate(self):
        # Both EM steps
        self.is_active = True
        self.is_eval = False

    def eval(self):
        # Only M step
        self.is_active = True
        self.is_eval = True

    def deactivate(self):
        # No EM step
        self.is_active = False

    def _init_centers(self):
        C = OrderedDict()
        for param in self.model.parameters():
            C[param] = self._init_centers_layer(param, self.nclusters)
        self.centers = C

    def _init_centers_layer(self, param, nclusters):
        centers = []
        if param.dim() == 1:
            # Biases
            dout = param.shape[0]
            centers = torch.rand((nclusters, dout)).cuda()/dout
        elif param.dim() == 2:
            # FC weights
            din, dout = param.shape
            centers = [
                torch.rand((nclusters, dout)).cuda()/(din+dout),
                torch.rand((nclusters, din)).cuda()/(din+dout)]
        else:
            # Convolution weights
            din = list(param.shape)[1:]
            dout = param.shape[0]
            centers = [
                torch.rand([nclusters]+din).cuda()/(din+dout),
                torch.rand((nclusters, dout)).cuda()/(din+dout)]
        return centers

    def _register_hooks(self):
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._em_step)

    def _save_input(self, module, input):
        if not self.is_active:
            return
        self.inputs[module] = input[0].clone().detach()

    def _em_step(self, module, grad_input, grad_output):
        if not self.is_active:
            return
        Ai = self.inputs[module]
        Gi = grad_input[0].clone().detach()
        Go = grad_output[0].clone().detach()
        self.ograds[module] = Go
        # print('%s %s' % (Ai.shape, Ai.shape))
        self.layer_cluster[module.__class__.__name__](module, Ai, Gi, Go)

    def _linear_cluster(self, module, Ai, Gi, Go):
        if not self.is_active:
            return
        # Bias
        Cb = self.centers[module.bias]
        # W^Ta_i + b = a_{i+1}
        # dL/db = dL/da_{i+1} da_{i+1}/db = sum(dL/da_{i+1})
        # Cb: C x d_out
        # Go: B x d_out
        # Ai: B x d_in
        # https://discuss.pytorch.org/t/dot-product-batch-wise/9746/3
        # CC = torch.matmul(Cb.unsqueeze(1), Cb.unsqueeze(2))
        # bmm is probably slower
        CC = (Cb*Cb).sum(1).unsqueeze(-1)
        assert CC.shape == (self.nclusters, 1), 'CC: C x 1.'
        CG = torch.matmul(Cb, Go.t())
        assert CG.shape == (self.nclusters, Ai.shape[0]), 'CG: C x B.'
        O = CC-2*CG
        self.batch_dist += [O]

        # Weight
        Ci, Co = self.centers[module.weight]
        CiAi = torch.matmul(Ci, Ai.t())
        CoGo = torch.matmul(Co, Go.t())
        CG = (CiAi)*(CoGo)
        assert CG.shape == (self.nclusters, Ai.shape[0]), 'CG: C x B.'
        CiCi = (Ci*Ci).view(Ci.shape[0], -1).sum(1)
        CoCo = (Co*Co).view(Co.shape[0], -1).sum(1)
        CC = ((CiCi)*(CoCo)).unsqueeze(-1)
        assert CC.shape == (self.nclusters, 1), 'CC: C x 1.'
        O = CC-2*CG
        assert O.shape == (self.nclusters, Ai.shape[0]), 'O: C x B.'

        # TODO: per-layer clusters
        self.batch_dist += [O]

    def _conv2d_cluster(self, module, a_in, d_in, d_out):
        if not self.is_active:
            return
        # TODO: conv
        pass

    def zero(self):
        if not self.is_active:
            return
        self.batch_dist = []
        self.inputs = {}
        self.ograds = {}

    def em_update(self):
        if not self.is_active:
            return
        # M: assign input
        # M: a_i = argmin_c(|g_i-C_c|^2)
        #        = argmin_c gg+CC-2*gC
        #        = argmin_c CC-2gC
        total_dist = torch.stack(self.batch_dist).sum(0)
        _, assign_i = total_dist.min(0)

        if self.is_eval:
            return assign_i

        # E: update C
        # E: C_c = mean_i(g_i * I(c==a_i))
        beta = self.beta
        assign_i = assign_i.unsqueeze(1)
        counts = torch.zeros(self.nclusters, 1).cuda()
        counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        self.cluster_size.mul_(beta).add_(counts)  # no *(1-beta)
        pre_size = self.cluster_size-counts
        # Reinit if no data is assigned to the cluster for some time
        # (time = beta decay).
        # If we reinit based only on the current batch assignemnts, it ignores
        # reinits constantly for the 5-example test
        reinits = (self.cluster_size < 1)
        nreinits = reinits.sum()
        self.reinits += reinits.long()

        # reinit from data
        perm = torch.randperm(self.inputs.values()[0].shape[0])[:nreinits]
        for module in self.inputs.keys():
            Ai = self.inputs[module]
            Go = self.ograds[module]

            # bias
            Cb = self.centers[module.bias]
            Cb_new = torch.zeros_like(Cb)
            Cb_new.scatter_add_(0, assign_i.expand_as(Go), Go)
            Cb.mul_(pre_size).add_(Cb_new).div_(self.cluster_size)

            # weight
            Ci, Co = self.centers[module.weight]
            Ci_new = torch.zeros_like(Ci)
            Co_new = torch.zeros_like(Co)
            # TODO: SVD
            Ci_new.scatter_add_(0, assign_i.expand_as(Ai), Ai)
            Co_new.scatter_add_(0, assign_i.expand_as(Go), Go)
            # Update clusters using the size
            Ci.mul_(pre_size).add_(Ci_new).div_(self.cluster_size)
            Co.mul_(pre_size).add_(Co_new).div_(self.cluster_size)

            # reinit centers
            if nreinits > 0:
                # Ci0, Co0 = self._init_centers_layer(module.weight, nzeros)
                # Ci.masked_scatter_(counts == 0, Ci0)
                # Co.masked_scatter_(counts == 0, Co0)

                # reinit from data
                Ci.masked_scatter_(reinits, Ai[perm])
                Co.masked_scatter_(reinits, Go[perm])
                Cb.masked_scatter_(reinits, Go[perm])
                self.cluster_size[reinits] = 1
        return assign_i

    def get_centers(self):
        centers = []
        for param, value in self.centers.iteritems():
            if param.dim() == 1:
                Cb = value
                centers += [Cb]
            elif param.dim() == 2:
                (Ci, Co) = value
                Cf = torch.matmul(Co.unsqueeze(-1), Ci.unsqueeze(1))
                centers += [Cf]
                assert Cf.shape == (Ci.shape[0], Co.shape[1], Ci.shape[1]),\
                    'Cf: C x d_out x d_in'
        return centers
