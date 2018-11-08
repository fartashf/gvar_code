import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from collections import OrderedDict
sys.path.append('../')
from models.mnist import MLP, MNISTNet  # NOQA


def get_gluster(model, opt):
    gluster_class = {'batch': GradientClusterBatch}
    gluster_args = {'nclusters': opt.gluster_num,
                    'beta': opt.gluster_beta
                    }
    return gluster_class[opt.gluster](model, **gluster_args)


class GradientCluster(object):
    def __init__(self, model, nclusters=1):
        # Q: duplicates
        # TODO: challenge: how many C? memory? time?
        self.layer_dist = {'Linear': self._linear_dist,
                           'Conv2d': self._conv2d_dist}
        self.known_modules = self.layer_dist.keys()

        self.is_active = True
        self.is_eval = False
        self.nclusters = nclusters
        # self.train_size = train_size
        self.model = model
        self.modules = []
        self.zero_data()
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

    def _init_centers(self, eps=1):
        C = OrderedDict()
        for param in self.model.parameters():
            C[param] = self._init_centers_layer(param, self.nclusters, eps)
        self.centers = C
        nclusters = self.nclusters
        self.cluster_size = torch.zeros(nclusters, 1).cuda()
        self.reinits = torch.zeros(nclusters, 1).long().cuda()

    def _init_centers_layer(self, param, nclusters, eps):
        centers = []
        if param.dim() == 1:
            # Biases
            dout = param.shape[0]
            centers = torch.rand((nclusters, dout)).cuda()/dout*eps
        elif param.dim() == 2:
            # FC weights
            din, dout = param.shape
            centers = [
                torch.rand((nclusters, dout)).cuda()/(din+dout)*eps,
                torch.rand((nclusters, din)).cuda()/(din+dout)*eps]
        else:
            # Convolution weights
            din = list(param.shape)[1:]
            dout = param.shape[0]
            centers = [
                torch.rand([nclusters]+din).cuda()/(din+dout)*eps,
                torch.rand((nclusters, dout)).cuda()/(din+dout)*eps]
        return centers

    def _register_hooks(self):
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_dists)

    def _save_input(self, module, input):
        if not self.is_active:
            return
        self.inputs[module] = input[0].clone().detach()

    def _save_dists(self, module, grad_input, grad_output):
        if not self.is_active:
            return
        Ai = self.inputs[module]
        Gi = grad_input[0].clone().detach()
        Go = grad_output[0].clone().detach()
        self.ograds[module] = Go
        # print('%s %s' % (Ai.shape, Ai.shape))
        self.layer_dist[module.__class__.__name__](module, Ai, Gi, Go)

    def _linear_dist(self, module, Ai, Gi, Go):
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

    def _conv2d_dist(self, module, a_in, d_in, d_out):
        if not self.is_active:
            return
        # TODO: conv
        pass

    def zero_data(self):
        if not self.is_active:
            return
        self.batch_dist = []
        self.inputs = {}
        self.ograds = {}

    def assign(self):
        raise NotImplemented('assign not implemented')

    def update(self, assign_i):
        raise NotImplemented('update not implemented')

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


class GradientClusterBatch(GradientCluster):
    def __init__(self, model, min_size, nclusters=1):
        super(GradientClusterBatch, self).__init__(model, nclusters)
        self.min_size = min_size

    def assign(self):
        """Assign inputs to clusters
        M: assign input
        M: a_i = argmin_c(|g_i-C_c|^2)
        = argmin_c gg+CC-2*gC
        = argmin_c CC-2gC
        """
        total_dist = torch.stack(self.batch_dist).sum(0)
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        counts = torch.zeros(self.nclusters, 1).cuda()
        counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        self.cluster_size.add_(counts)
        return assign_i, batch_dist

    def update_batch(self, data_loader, device='cuda'):
        # TODO: read through and write a simple test
        model = self.model
        assign_i = torch.zeros(len(data_loader), 1).long().to(device)
        total_dist = torch.zeros(self.nclusters, 1).to(device)
        self.cluster_size = torch.zeros(self.nclusters, 1).cuda()

        # for all data save dists
        # for all data assign
        for data, target, idx in data_loader:
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            self.zero_data()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            ai, batch_dist = self.assign()
            assign_i[idx] = ai
            total_dist.scatter_add_(0, ai, batch_dist)

        self._init_centers(0)
        total_dist.div_(self.cluster_size)

        # split the scattered cluster if the smallest cluster is small
        # TODO: more than one split, challenge: don't know the new dist
        # after one split
        s, i = self.cluster_size.min(0)
        if s < self.min_size:
            _, j = total_dist.max(0)
            idx = torch.arange(assign_i.shape[0])[assign_i[:, 0] == j]
            assign_i[idx[torch.randperm(len(idx))[:len(idx)/2]]] = i

        # for all data update centers
        for data, target, idx in data_loader:
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            self.zero_data()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            assign_i[idx], _ = self.assign()  # recompute A, G
            self.update(assign_i[idx])

        self.normalize()
        return assign_i

    def update(self, assign_i):
        """Update Clusters
        E: update C
        E: C_c = mean_i(g_i * I(c==a_i))
        """

        for module in self.inputs.keys():
            Ai = self.inputs[module]
            Go = self.ograds[module]

            # bias
            Cb = self.centers[module.bias]
            Cb_new = torch.zeros_like(Cb)
            Cb_new.scatter_add_(0, assign_i.expand_as(Go), Go)
            Cb.add_(Cb_new)

            # weight
            Ci, Co = self.centers[module.weight]
            Ci_new = torch.zeros_like(Ci)
            Co_new = torch.zeros_like(Co)
            # TODO: SVD
            Ci_new.scatter_add_(0, assign_i.expand_as(Ai), Ai)
            Co_new.scatter_add_(0, assign_i.expand_as(Go), Go)
            # Update clusters
            Ci.add_(Ci_new)
            Co.add_(Co_new)

    def normalize(self):
        for module in self.inputs.keys():
            Cb = self.centers[module.bias]
            Cb.div_(self.cluster_size)

            Ci, Co = self.centers[module.weight]
            Ci.div_(self.cluster_size)
            Co.div_(self.cluster_size)


class GradientClusterOnline(GradientCluster):
    def __init__(self, model, beta=0.9, nclusters=1):
        super(GradientClusterOnline, self).__init__(model, nclusters)
        self.beta = beta  # cluster size decay factor

    def em_step(self):
        if self.is_active:
            assign_i = self.assign()
            if not self.is_eval:
                self.update(assign_i)
            return assign_i

    def assign(self):
        # M: assign input
        # M: a_i = argmin_c(|g_i-C_c|^2)
        #        = argmin_c gg+CC-2*gC
        #        = argmin_c CC-2gC
        total_dist = torch.stack(self.batch_dist).sum(0)
        _, assign_i = total_dist.min(0)
        return assign_i

    def update(self, assign_i):
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
