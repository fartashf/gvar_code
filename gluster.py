import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from collections import OrderedDict
import numpy as np
sys.path.append('../')
from models.mnist import MLP, MNISTNet  # NOQA


def get_gluster(model, opt):
    gluster_class = {'batch': GradientClusterBatch}
    gluster_args = {'nclusters': opt.gluster_num,
                    'beta': opt.gluster_beta
                    }
    return gluster_class[opt.gluster](model, **gluster_args)


def get_gluster_module(module):
    module_name = module.__class__.__name__
    glayers = {'Linear': GlusterLinear, 'Conv2D': GlusterConv}
    if module_name in glayers:
        return glayers[module_name]
    if len(module.children()) > 0:
        return GlusterModuleMulti
    return None


class GlusterModule(object):
    def __init__(self, module, eps, nclusters, no_grad=False, **kwargs):
        self.module = module
        self.C = nclusters
        self.eps = eps
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        module.register_forward_pre_hook(self._save_input_hook)
        module.register_backward_hook(self._save_dist_hook)
        self.is_active = True
        self.is_eval = False
        self.no_grad = no_grad  # grad=1 => clustring in the input space

    def _save_input_hook(self, Ai):
        if not self.is_active:
            return
        self.Ai = Ai[0].clone().detach()

    def _save_dist_hook(self, grad_input, grad_output):
        if not self.is_active:
            return
        Ai = self.Ai
        Gi = grad_input[0].clone().detach()
        Go = grad_output[0].clone().detach()
        self.Gi = Gi
        self.Go = Go
        if self.no_grad:
            Gi.fill_(1./Gi.numel())
            Go.fill_(1./Go.numel())
        self.ograds = Go
        # print('%s %s' % (Ai.shape, Ai.shape))
        O = []
        if self.module.bias:
            O += [self._save_dist_hook_bias(Ai, Gi, Go)]
        if self.module.weight:
            O += [self._save_dist_hook_weight(Ai, Gi, Go)]
        # TODO: per-layer clusters
        self.batch_dist += O

    def __zero_new_centers(self):
        if self.module.bias:
            self._zero_new_centers_bias()
        if self.module.weight:
            self._zero_new_centers_weight()

    def __accum_new_centers(self, assign_i):
        if self.module.bias:
            self._accum_new_centers_bias(assign_i)
        if self.module.weight:
            self._accum_new_centers_weight(assign_i)

    def __update_batch(self, cluster_size):
        if self.module.bias:
            self._update_batch_bias(cluster_size)
        if self.module.weight:
            self._update_batch_weight(cluster_size)

    def __update_online(self, pre_size, new_size):
        if self.module.bias:
            self._update_online_bias(pre_size, new_size)
        if self.module.weight:
            self._update_online_weight(pre_size, new_size)

    def get_dist(self):
        # called after a backward pass with hooks
        return self.batch_dist

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


class GlusterLinear(GlusterModule):
    def __init__(self, **kwargs):
        super(GlusterLinear, self).__init__(self, **kwargs)
        weight = self.module.weight
        din, dout = weight.shape
        dout = weight.shape[0]
        # TODO: change init to 2/din
        self.Ci = torch.rand((self.C, dout)).cuda()/(din+dout)*self.eps
        self.Co = torch.rand((self.C, din)).cuda()/(din+dout)*self.eps
        self.Cb = torch.rand((self.C, dout)).cuda()/dout*self.eps
        self.Ci_new = torch.zeros_like(self.Ci)
        self.Co_new = torch.zeros_like(self.Co)
        self.Cb_new = torch.zeros_like(self.Cb)

    def _save_dist_hook_bias(self, Ai, Gi, Go):
        C = self.nclusters
        B = Ai.shape[0]

        Cb = self.Cb
        # W^Ta_i + b = a_{i+1}
        # dL/db = dL/da_{i+1} da_{i+1}/db = sum(dL/da_{i+1})
        # Cb: C x d_out
        # Go: B x d_out
        # Ai: B x d_in
        # https://discuss.pytorch.org/t/dot-product-batch-wise/9746/3
        # CC = torch.matmul(Cb.unsqueeze(1), Cb.unsqueeze(2))
        # bmm is probably slower
        CC = (Cb*Cb).sum(1).unsqueeze(-1)
        assert CC.shape == (C, 1), 'CC: C x 1.'
        CG = torch.matmul(Cb, Go.t())
        assert CG.shape == (C, B), 'CG: C x B.'
        O = CC-2*CG
        return O

    def _save_dist_hook_weight(self, Ai, Gi, Go):
        O = []
        C = self.nclusters
        B = Ai.shape[0]

        Ci, Co = self.Ci, self.Cb
        CiAi = torch.matmul(Ci, Ai.t())
        CoGo = torch.matmul(Co, Go.t())
        CG = (CiAi)*(CoGo)
        assert CG.shape == (C, B), 'CG: C x B.'
        CiCi = (Ci*Ci).view(Ci.shape[0], -1).sum(1)
        CoCo = (Co*Co).view(Co.shape[0], -1).sum(1)
        CC = ((CiCi)*(CoCo)).unsqueeze(-1)
        assert CC.shape == (C, 1), 'CC: C x 1.'
        O += [CC-2*CG]
        assert O.shape == (C, B), 'O: C x B.'
        return O

    def zero_new_centers(self):
        # TODO: figure out when zeroing should be done with batch/online
        self.Ci_new.fill_(0)
        self.Co_new.fill_(0)
        self.Cb_new.fill_(0)

    def accum_new_centers(self, assign_i):
        # TODO: SVD
        self.Ci_new.scatter_add_(0, assign_i.expand_as(self.Ai), self.Ai)
        self.Co_new.scatter_add_(0, assign_i.expand_as(self.Go), self.Go)
        self.Cb_new.scatter_add_(0, assign_i.expand_as(self.Go), self.Go)

    def update_batch(self, cluster_size):
        self.Ci.copy_(self.Ci_new).div_(cluster_size.clamp(1))
        self.Co.copy_(self.Co_new).div_(cluster_size.clamp(1))
        self.Cb.copy_(self.Cb_new).div_(cluster_size.clamp(1))

    def _update_online_bias(self, pre_size, new_size):
        self.Cb.mul_(pre_size).add_(self.Cb_new).div_(new_size)

    def _update_online_weight(self, pre_size, new_size):
        self.Ci.mul_(pre_size).add_(self.Ci_new).div_(new_size)
        self.Co.mul_(pre_size).add_(self.Co_new).div_(new_size)

    def reinit_from_data(self, reinits, perm):
        # Ci0, Co0 = self._init_centers_layer(module.weight, nzeros)
        # Ci.masked_scatter_(counts == 0, Ci0)
        # Co.masked_scatter_(counts == 0, Co0)
        self.Ci.masked_scatter_(reinits, self.Ai[perm])
        self.Co.masked_scatter_(reinits, self.Go[perm])
        self.Cb.masked_scatter_(reinits, self.Go[perm])

    def reinit_from_largest(self, ri, li):
        """
        reinit from the largest cluster
        reinit one at a time
        """
        # reinit centers
        # X[ri] = is index_put_(ri, ...)
        # TODO: figure out autograd with this copy
        self.Ci[ri] = self.Ci[li]
        self.Co[ri] = self.Co[li]
        self.Cb[ri] = self.Cb[li]
        # Adding noise did not help on the first test
        # Don't do this. Ruins assignments if accidentally reinits
        # Ci.index_add_(0, ri, torch.rand_like(Ci[ri])*self.total_dist[li])
        # Co.index_add_(0, ri, torch.rand_like(Co[ri])*self.total_dist[li])
        # Cb.index_add_(0, ri, torch.rand_like(Cb[ri])*self.total_dist[li])


class GlusterConv(GlusterModule):
    def __init__(self, **kwargs):
        super(GlusterConv, self).__init__(self, **kwargs)
        self.param = self.module.weight
        din = list(self.param.shape)[1:]
        dout = self.param.shape[0]
        eps = self.eps
        self.Ci = torch.rand([self.C]+din).cuda()/(np.prod(din)+dout)*eps,
        self.Co = torch.rand((self.C, dout)).cuda()/(np.prod(din)+dout)*eps
        self.Cb = torch.rand((self.C, dout)).cuda()/dout*self.eps

    def _save_dist_hook_bias(self, Ai0, Gi, Go0):
        module = self.module
        C = self.nclusters
        B = Ai0.shape[0]
        Go = Go0.reshape(Go0.shape[:-2]+(np.prod(Go0.shape[-2:]), ))

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
        assert CC.shape == (C, 1), 'CC: C x 1.'
        CG = torch.einsum('ko,bot->kb', [Cb, Go])
        assert CG.shape == (C, B), 'CG: C x B.'
        O = CC-2*CG
        return O

    def _save_dist_hook_weight(self, Ai0, Gi, Go0):
        module = self.module
        param = self.param
        C = self.nclusters
        B = Ai0.shape[0]
        din = param.shape[1:]
        dout = param.shape[0]
        Go = Go0.reshape(Go0.shape[:-2]+(np.prod(Go0.shape[-2:]), ))
        T = Go.shape[-1]

        assert Go.shape == (B, dout, T), 'Go: B x dout x T'
        Ai = F.unfold(
                Ai0, module.kernel_size,
                padding=module.padding,
                stride=module.stride, dilation=module.dilation)
        assert Ai.shape == (B, np.prod(din), T), 'Ai: B x din x T'
        Ci, Co = self.centers[module.weight]
        assert Ci.shape == (C, )+din, 'Ci: C x din'
        assert Co.shape == (C, dout), 'Co: C x dout'
        Ci = Ci.reshape((C, -1))
        # TODO: one big einsum
        CiAi = torch.einsum('ki,bit->kbt', [Ci, Ai])
        CoGo = torch.einsum('ko,bot->kbt', [Co, Go])
        CG = torch.einsum('kbt,kbt->kb', [CiAi, CoGo])
        assert CG.shape == (C, B), 'CG: C x B.'
        CiCi = (Ci*Ci).view(Ci.shape[0], -1).sum(1)
        CoCo = (Co*Co).view(Co.shape[0], -1).sum(1)
        CC = ((CiCi)*(CoCo)).unsqueeze(-1)
        assert CC.shape == (C, 1), 'CC: C x 1.'
        O = CC-2*CG
        assert O.shape == (C, B), 'O: C x B.'

        return O


class GlusterModuleMulti(GlusterModule):
    def __init__(
            self, module, eps, nclusters, no_grad=False,
            ignore_modules=[]):
        self.module = module
        self.ignore_modules = ignore_modules
        self.C = nclusters
        self.eps = eps
        self.G = OrderedDict()
        for name, m in module.named_modules():
            if name in self.ignore_modules:
                continue
            glayer = get_gluster_module(m)
            self.G[m] = glayer(m, eps, nclusters)

    def get_dist(self):
        batch_dist = []
        for module in self.G.keys():
            batch_dist += self.G[module].get_dist()
        return batch_dist

    def update_batch(self, assign_i):
        """Update Clusters
        E: update C
        E: C_c = mean_i(g_i * I(c==a_i))
        """
        for module in self.G.keys():
            self.G[module].update_batch(assign_i)

    def update_online(self, assign_i):
        """Update Clusters
        E: update C
        E: C_c = mean_i(g_i * I(c==a_i))
        """
        for module in self.G.keys():
            self.G[module].update_online(assign_i)

    def normalize(self):
        for module in self.G.keys():
            self.G[module].normalize()


class GradientCluster(object):
    def __init__(
            self, model, nclusters=1, no_grad=False,
            ignore_modules=[], **kwargs):
        # Q: duplicates
        # TODO: challenge: how many C? memory? time?

        self.nclusters = nclusters
        self.model = model
        self.device = device = self.model.parameters().next().device
        # self.zero_data()
        # self._init_centers()
        self.reinits = torch.zeros(nclusters, 1).long().to(device)
        self.cluster_size = torch.zeros(nclusters, 1).to(device)
        self.total_dist = torch.zeros(self.nclusters, 1).to(device)

        self.G = GlusterModuleMulti(
                model, 1, nclusters, no_grad=no_grad,
                ignore_modules=ignore_modules)

    def activate(self):
        self.G.activate()

    def eval(self):
        self.G.eval()

    def deactivate(self):
        self.G.deactivate()

    def copy_(self, model):
        for m, g in zip(model.parameters(), self.model.parameters()):
            g.data.copy_(m.data)

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

    def print_stats(self):
        centers = self.get_centers()
        nclusters = centers[0].shape[0]

        normC = np.zeros((nclusters,))
        L = [c.pow(2).view(nclusters, -1).sum(1).cpu().numpy()
             for c in centers]
        normC = np.sqrt(np.sum(L, 0))
        print('normC:')
        print(normC)
        print('Reinit count: %s' % str(self.reinits.cpu().numpy()))
        print('Cluster size: %s' % str(self.cluster_size.cpu().numpy()))
        print('Total dists: %s' % str(self.total_dist.cpu().numpy()))
        return normC


class GradientClusterBatch(GradientCluster):
    def __init__(self, model, min_size, **kwargs):
        super(GradientClusterBatch, self).__init__(model, **kwargs)
        self.min_size = min_size

    def assign(self):
        """Assign inputs to clusters
        M: assign input
        M: a_i = argmin_c(|g_i-C_c|^2)
        = argmin_c gg+CC-2*gC
        = argmin_c CC-2gC
        """
        batch_dist = self.G.get_dist()
        total_dist = torch.stack(batch_dist).sum(0)
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        if not self.is_eval:
            counts = torch.zeros(self.nclusters, 1).cuda()
            counts.scatter_add_(0, assign_i, torch.ones_like(assign_i).float())
            self.cluster_size.add_(counts)
        return assign_i, batch_dist

    def update_batch(self, data_loader, train_size):
        model = self.model
        device = self.device
        assign_i = torch.zeros(train_size, 1).long().to(device)
        pred_i = np.zeros(train_size)
        loss_i = np.zeros(train_size)
        target_i = np.zeros(train_size)
        total_dist = torch.zeros(self.nclusters, 1).to(device)
        self.cluster_size = torch.zeros(self.nclusters, 1).cuda()

        print('> Gluster batch: Save distortions and assign data to clusters.')
        for data, target, idx in data_loader:
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            # self.zero_data()
            output = self.model(data)
            pred_i[idx] = output.max(1, keepdim=True)[1].cpu().numpy()[:, 0]
            target_i[idx] = target.cpu().numpy()
            loss = F.nll_loss(output, target, reduction='none')
            loss_i[idx] = loss.detach().cpu().numpy()
            loss = loss.mean()
            loss.backward()
            ai, batch_dist = self.assign()
            assign_i[idx] = ai
            total_dist.scatter_add_(0, ai, batch_dist)
        total_dist.div_(self.cluster_size.clamp(1))
        td = total_dist.sum().item()
        self.total_dist.copy_(total_dist)
        print('> Gluster batch: total distortions: %f (negative is fine)' % td)

        # split the scattered cluster if the smallest cluster is small
        # More than one split: don't know the new dist after one split
        s, i = self.cluster_size.min(0)
        if s < self.min_size:
            # TODO: negative distortion
            _, j = total_dist.masked_fill(
                self.cluster_size < self.min_size, float('-inf')).max(0)
            idx = torch.arange(assign_i.shape[0])[assign_i[:, 0] == j]
            assign_i[idx[torch.randperm(len(idx))[:len(idx)/2]]] = i
            self.reinits[i] += 1

        print('> Gluster batch: Update cluster centers given the assignments')
        self._init_centers(0)
        self.G.zero_new_centers()
        for data, target, idx in data_loader:
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            # self.zero_data()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            # recompute A, G but don't do reassign as clusters change
            self.assign()
            # assign_i[idx], _ = self.assign()  # recompute A, G
            self.G.accum_new_centers(assign_i[idx])

        self.cluster_size.fill_(0)
        self.cluster_size.scatter_add_(0, assign_i,
                                       torch.ones_like(assign_i).float())
        self.G.update_batch(self.cluster_size)
        # td before E step, we have to do another loop for td after E step
        return td, assign_i.cpu().numpy(), target_i, pred_i, loss_i


class GradientClusterOnline(GradientCluster):
    def __init__(self, model, beta=0.9, min_size=1,
                 reinit_method='data', **kwargs):
        super(GradientClusterOnline, self).__init__(model, **kwargs)
        self.beta = beta  # cluster size decay factor
        self.min_size = 1
        self.reinit_method = reinit_method  # 'data' or 'largest'
        self.cluster_size.fill_(self.min_size)

    def em_step(self):
        with torch.no_grad():
            if self.is_active:
                assign_i, batch_dist = self.assign()
                if not self.is_eval:
                    invalid_clusters = self.update(assign_i, batch_dist)
                    return assign_i, batch_dist, invalid_clusters
                return assign_i, batch_dist

    def assign(self):
        # M: assign input
        # M: a_i = argmin_c(|g_i-C_c|^2)
        #        = argmin_c gg+CC-2*gC
        #        = argmin_c CC-2gC
        batch_dist = self.G.get_dist()
        total_dist = torch.stack(batch_dist).sum(0)
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        return assign_i, batch_dist

    def update(self, assign_i, batch_dist):
        # Not keeping a full internal assignment list
        # E: update C
        # E: C_c = mean_i(g_i * I(c==a_i))
        beta = self.beta
        counts = torch.zeros(self.nclusters, 1).cuda()
        counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        self.cluster_size.mul_(beta).add_(counts)  # no *(1-beta)
        pre_size = self.cluster_size-counts
        self.total_dist.mul_(pre_size).scatter_add_(
            0, assign_i, batch_dist).div_(self.cluster_size)

        self.G.zero_new_centers()
        self.G.accum_new_centers(assign_i)
        self.G.update_online(pre_size, self.cluster_size)

        invalid_clusters = self.reinit()
        return invalid_clusters

    def reinit(self):
        # Reinit if no data is assigned to the cluster for some time
        # (time = beta decay).
        # If we reinit based only on the current batch assignemnts, it ignores
        # reinits constantly for the 5-example test
        reinits = (self.cluster_size < self.min_size)
        nreinits = reinits.sum()
        self.reinits += reinits.long()
        # TODO: stop reinit or delay if too often
        if nreinits > 0:
            if self.reinit_method == 'data':
                return self.reinit_from_data(reinits)
            elif self.reinit_method == 'largest':
                return self.reinit_from_largest(reinits)
            else:
                raise Exception('Reinit method not defined.')
        return []

    def reinit_from_data(self, reinits):
        nreinits = reinits.sum()
        # reinit from data
        perm = torch.randperm(self.inputs.values()[0].shape[0])[:nreinits]
        self.cluster_size[reinits] = 1
        self.G.reinit_from_data(reinits, perm)
        invalid_clusters = reinits.cpu().numpy()
        return invalid_clusters

    def reinit_from_largest(self, reinits):
        # reinit from the largest cluster
        # reinit one at a time
        _, ri = reinits.max(0)
        tdm = self.total_dist.masked_fill(
            self.cluster_size < self.min_size, float('-inf'))
        # TODO: Ties should be broken deterministically
        _, li = tdm.max(0)
        self.G.reinit_from_largest(ri, li)
        self.cluster_size[ri] = self.cluster_size[li] / 2
        self.cluster_size[li] = self.cluster_size[li] / 2
        # This is not exact, maybe add a multiple of cluster size
        self.total_dist[ri] = self.total_dist[li]  # -1e-5
        # print(self.cluster_size)
        invalid_clusters = [ri]
        return invalid_clusters


class GradientClusterOnlineMemory(GradientCluster):
    """
    Main issue: when should we reset the cluster size?
    """
    def __init__(self, model, min_size, train_size, **kwargs):
        super(GradientClusterOnlineMemory, self).__init__(model, **kwargs)
        self.train_size = train_size
        self.assign_i = torch.zeros(train_size, 1).long().to(self.device)

    def em_step(self):
        if self.is_active:
            assign_i = self.assign()
            if not self.is_eval:
                self.update(assign_i)
            return assign_i

    def assign(self, idx):
        """Assign inputs to clusters
        M: assign input
        M: a_i = argmin_c(|g_i-C_c|^2)
        = argmin_c gg+CC-2*gC
        = argmin_c CC-2gC
        """
        batch_dist = self.G.get_dist()
        total_dist = torch.stack(batch_dist).sum(0)
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        return assign_i, batch_dist

    def update(self, assign_i, batch_dist, idx):
        # E: update C
        # E: C_c = mean_i(g_i * I(c==a_i))
        # Keep a full internal assignment list
        self.assign_i[idx] = assign_i
        assign_i = assign_i.unsqueeze(1)

        # Keep an exact cluster size given the internal assignment
        # TODO: is this the only place we use self.assign_i?
        self.cluster_size.fill_(0)
        self.cluster_size.scatter_add_(0, assign_i,
                                       torch.ones_like(assign_i).float())
        counts = torch.zeros(self.nclusters, 1).cuda()
        counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        pre_size = self.cluster_size-counts

        # Reinit if cluster size small
        reinits = (self.cluster_size < self.min_size)
        nreinits = reinits.sum()
        self.reinits += reinits.long()

        for module in self.inputs.keys():
            Ai = self.inputs[module]
            Go = self.ograds[module]

            # TODO: refactor copies of this
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
                # TODO: ?reinit from the largest cluster?
                pass
