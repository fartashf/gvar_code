import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


def get_gluster_module(module):
    module_name = module.__class__.__name__
    glayers = {
            'Sequential': GlusterModule,
            'Linear': GlusterLinear, 'Conv2D': GlusterConv}
    if module_name in glayers:
        return glayers[module_name]
    return None


class GlusterModule(object):
    def __init__(
            self, module, eps, nclusters, no_grad=False,
            ignore_modules=[], *args, **kwargs):
        self.module = module
        self.nclusters = nclusters
        self.eps = eps
        self.batch_dist = None
        self.is_active = True
        self.is_eval = False
        self.no_grad = no_grad  # grad=1 => clustring in the input space
        self.ignore_modules = ignore_modules
        # TODO: unnamed children
        self.active_modules = [
                m for name, m in module.named_children() if name
                not in ignore_modules]
        self.children = OrderedDict()
        self._register_hooks()

    def _register_hooks(self):
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        self.module.register_forward_pre_hook(self._save_input_hook)
        self.module.register_backward_hook(self._save_dist_hook)

    def _save_input_hook(self, m, Ai):
        if not self.is_active:
            return
        self.Ai = Ai[0].clone().detach()

    def _save_dist_hook(self, m, grad_input, grad_output):
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
        if self.module.bias is not None:
            O += [self._save_dist_hook_bias(Ai, Gi, Go)]
        if self.module.weight is not None:
            O += [self._save_dist_hook_weight(Ai, Gi, Go)]
        # TODO: per-layer clusters
        self.batch_dist = O

    def get_dist(self):
        # called after a backward
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

    def update_online(self, pre_size, new_size):
        self.Ci.mul_(pre_size).add_(self.Ci_new).div_(new_size)
        self.Co.mul_(pre_size).add_(self.Co_new).div_(new_size)
        self.Cb.mul_(pre_size).add_(self.Cb_new).div_(new_size)

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

    def get_centers(self):
        centers = []
        centers += [self.Cb]
        Cf = torch.matmul(self.Co.unsqueeze(-1), self.Ci.unsqueeze(1))
        centers += [Cf]
        assert Cf.shape == (
                self.Ci.shape[0], self.Co.shape[1], self.Ci.shape[1]
                ), 'Cf: C x d_out x d_in'
        return centers


class LoopFunction(object):
    def __init__(self, children, fname):
        fs = []
        for module in children.keys():
            fs += [getattr(children[module], fname)]
        self.fs = fs

    def __call__(self, *args, **kwargs):
        rs = []
        for f in self.fs:
            r = f(*args, **kwargs)
            if r is not None:
                rs += r
        return rs


class GlusterContainer(GlusterModule):
    def __init__(self, *args, **kwargs):
        super(GlusterContainer, self).__init__(*args, **kwargs)
        self._init_children(*args, **kwargs)
        self._set_default_loop()
        print('Gluster init done.')

    def _register_hooks(self):
        pass

    def _init_children(self, *args, **kwargs):
        for m in self.active_modules:
            glayer = get_gluster_module(m)
            self.children[m] = glayer(*((m, )+args[1:]), **kwargs)

    def _set_default_loop(self):
        loop_functions = [
                'update_batch', 'update_online', 'zero_new_centers',
                'accum_new_centers', 'reinit_from_data',
                'reinit_from_largest', 'get_dist', 'get_centers']
        for fname in loop_functions:
            setattr(self, fname, LoopFunction(self.children, fname))


class GlusterLinear(GlusterModule):
    def __init__(self, *args, **kwargs):
        super(GlusterLinear, self).__init__(*args, **kwargs)
        weight = self.module.weight
        dout, din = weight.shape
        # TODO: change init to 2/din
        C = self.nclusters
        self.Ci = torch.rand((C, din)).cuda()/(din+dout)*self.eps
        self.Co = torch.rand((C, dout)).cuda()/(din+dout)*self.eps
        self.Cb = torch.rand((C, dout)).cuda()/dout*self.eps
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
        O = CC-2*CG
        assert O.shape == (C, B), 'O: C x B.'
        return O


class GlusterConv(GlusterModule):
    def __init__(self, *args, **kwargs):
        super(GlusterConv, self).__init__(*args, **kwargs)
        self.param = self.module.weight
        din = list(self.param.shape)[1:]
        dout = self.param.shape[0]
        eps = self.eps
        C = self.nclusters
        self.Ci = torch.rand([C]+din).cuda()/(np.prod(din)+dout)*eps,
        self.Co = torch.rand((C, dout)).cuda()/(np.prod(din)+dout)*eps
        self.Cb = torch.rand((C, dout)).cuda()/dout*self.eps

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
