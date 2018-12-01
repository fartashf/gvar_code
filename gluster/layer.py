import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


def get_gluster_module(module):
    module_name = module.__class__.__name__
    glayers = {
            'Sequential': GlusterModule,
            'Linear': GlusterLinear, 'Conv2d': GlusterConv}
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
        self.has_param = (hasattr(module, 'weight') or hasattr(module, 'bias'))
        self._register_hooks()

    def _register_hooks(self):
        if not self.has_param:
            return
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        self.module.register_forward_pre_hook(self._save_input_hook)
        self.module.register_backward_hook(self._save_dist_hook)

    def _save_input_hook(self, m, Ai):
        if not self.is_active:
            return
        if not self.has_param:
            return
        Ai = Ai[0].clone().detach()
        self.Ai = self._post_proc_Ai(Ai)

    def _post_proc_Ai(self, Ai0):
        return Ai0

    def _post_proc_Go(self, Go0):
        if self.no_grad:
            Go0.fill_(1./Go0.numel())
        return Go0

    def _save_dist_hook(self, m, grad_input, grad_output):
        if not self.is_active:
            return
        if not self.has_param:
            return
        Ai = self.Ai
        # Gi = grad_input[0].clone().detach()  # has 3 elements, 1 is Gi?
        Go0 = grad_output[0].clone().detach()
        self.Go = Go = self._post_proc_Go(Go0)
        # print('%s %s' % (Ai.shape, Ai.shape))
        O = []
        if self.module.bias is not None:
            O += [self._save_dist_hook_bias(Ai, Go)]
        if self.module.weight is not None:
            O += [self._save_dist_hook_weight(Ai, Go)]
        # TODO: per-layer clusters
        self.batch_dist = O

    def get_dist(self):
        if not self.has_param:
            return
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
        if not self.has_param:
            return
        # TODO: figure out when zeroing should be done with batch/online
        self.Ci_new.fill_(0)
        self.Co_new.fill_(0)
        self.Cb_new.fill_(0)

    def update_batch(self, cluster_size):
        if not self.has_param:
            return
        self.Ci.copy_(self.Ci_new).div_(cluster_size.clamp(1))
        self.Co.copy_(self.Co_new).div_(cluster_size.clamp(1))
        self.Cb.copy_(self.Cb_new).div_(cluster_size.clamp(1))

    def update_online(self, pre_size, new_size):
        if not self.has_param:
            return
        self.Ci.mul_(pre_size).add_(self.Ci_new).div_(new_size)
        self.Co.mul_(pre_size).add_(self.Co_new).div_(new_size)
        self.Cb.mul_(pre_size).add_(self.Cb_new).div_(new_size)

    def accum_new_centers(self, assign_i):
        if not self.has_param:
            return
        raise Exception('Not implemented.')

    def reinit_from_data(self, reinits, perm):
        if not self.has_param:
            return
        raise Exception('Not implemented.')

    def reinit_from_largest(self, ri, li):
        if not self.has_param:
            return
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
        if not self.has_param:
            return
        centers = []
        Cf = torch.matmul(self.Co.unsqueeze(-1), self.Ci.unsqueeze(1))
        centers += [Cf]
        centers += [self.Cb]
        assert Cf.shape == (
                self.Ci.shape[0], self.Co.shape[1], self.Ci.shape[1]
                ), 'Cf: C x d_out x d_in'
        return centers


class LoopFunction(object):
    def __init__(self, module, children, fname):
        fs = [getattr(super(module.__class__, module), fname)]
        for module in children.keys():
            fs += [getattr(children[module], fname)]
        self.fs = fs
        self.fname = fname

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
            if glayer is not None:
                self.children[m] = glayer(*((m, )+args[1:]), **kwargs)

    def _set_default_loop(self):
        loop_functions = [
                'update_batch', 'update_online', 'zero_new_centers',
                'accum_new_centers', 'reinit_from_data',
                'reinit_from_largest', 'get_dist', 'get_centers',
                'deactivate', 'activate', 'eval']
        for fname in loop_functions:
            setattr(self, fname, LoopFunction(self, self.children, fname))


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

    def _save_dist_hook_bias(self, Ai, Go):
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

    def _save_dist_hook_weight(self, Ai, Go):
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

    def reinit_from_data(self, reinits, perm):
        # Ci0, Co0 = self._init_centers_layer(module.weight, nzeros)
        # Ci.masked_scatter_(counts == 0, Ci0)
        # Co.masked_scatter_(counts == 0, Co0)
        self.Ci.masked_scatter_(reinits, self.Ai[perm])
        self.Co.masked_scatter_(reinits, self.Go[perm])
        self.Cb.masked_scatter_(reinits, self.Go[perm])

    def accum_new_centers(self, assign_i):
        # TODO: SVD
        Ai = self.Ai
        Go = self.Go
        self.Ci_new.scatter_add_(0, assign_i.expand_as(Ai), Ai)
        self.Co_new.scatter_add_(0, assign_i.expand_as(Go), Go)
        self.Cb_new.scatter_add_(0, assign_i.expand_as(Go), Go)


class GlusterConv(GlusterModule):
    def __init__(self, *args, **kwargs):
        super(GlusterConv, self).__init__(*args, **kwargs)
        self.param = self.module.weight
        din = np.prod(list(self.param.shape)[1:])
        dout = self.param.shape[0]
        eps = self.eps
        C = self.nclusters
        self.Ci = torch.rand((C, din)).cuda()/(din+dout)*eps
        self.Co = torch.rand((C, dout)).cuda()/(din+dout)*eps
        self.Cb = torch.rand((C, dout)).cuda()/dout*self.eps
        self.Ci_new = torch.zeros_like(self.Ci)
        self.Co_new = torch.zeros_like(self.Co)
        self.Cb_new = torch.zeros_like(self.Cb)

    def _post_proc_Ai(self, Ai0):
        module = self.module
        Ai = F.unfold(
                Ai0, module.kernel_size,
                padding=module.padding,
                stride=module.stride, dilation=module.dilation)
        return Ai

    def _post_proc_Go(self, Go0):
        Go = Go0.reshape(Go0.shape[:-2]+(np.prod(Go0.shape[-2:]), ))
        return Go

    def _save_dist_hook_bias(self, Ai, Go):
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
        CG = torch.einsum('ko,bot->kb', [Cb, Go])
        assert CG.shape == (C, B), 'CG: C x B.'
        O = CC-2*CG
        return O

    def _save_dist_hook_weight(self, Ai, Go):
        param = self.param
        C = self.nclusters
        B = Ai.shape[0]
        din = np.prod(param.shape[1:])
        dout = param.shape[0]
        T = Go.shape[-1]

        assert Go.shape == (B, dout, T), 'Go: B x dout x T'
        assert Ai.shape == (B, din, T), 'Ai: B x din x T'
        Ci, Co = self.Ci, self.Co
        assert Ci.shape == (C, din), 'Ci: C x din'
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

    def reinit_from_data(self, reinits, perm):
        Ai = self.Ai[perm].sum(-1)
        Go = self.Go[perm].sum(-1)
        self.Ci.masked_scatter_(reinits, Ai)
        self.Co.masked_scatter_(reinits, Go)
        self.Cb.masked_scatter_(reinits, Go)

    def accum_new_centers(self, assign_i):
        if not self.has_param:
            return
        # TODO: SVD
        Ai = self.Ai.sum(-1)
        Go = self.Go.sum(-1)
        self.Ci_new.scatter_add_(0, assign_i.expand_as(Ai), Ai)
        self.Co_new.scatter_add_(0, assign_i.expand_as(Go), Go)
        self.Cb_new.scatter_add_(0, assign_i.expand_as(Go), Go)
