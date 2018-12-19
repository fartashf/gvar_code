import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import logging


def get_gluster_module(module):
    module_name = module.__class__.__name__
    glayers = {'Linear': GlusterLinear, 'Conv2d': GlusterConv}
    if module_name in glayers:
        return glayers[module_name]
    if len(list(module.children())) > 0:
        return GlusterContainer
    return None


class GlusterModule(object):
    def __init__(
            self, module, eps, nclusters, no_grad=False,
            ignore_modules=[], active_only='', name='', *args, **kwargs):
        self.module = module
        self.nclusters = nclusters
        self.eps = eps
        self.batch_dist = None
        self.is_active = True
        self.is_eval = False
        self.no_grad = no_grad  # grad=1 => clustring in the input space
        self.ignore_modules = ignore_modules
        self.name = name
        self.active_only = active_only
        # TODO: unnamed children
        self.active_modules = [
                (n, m) for n, m in module.named_children()
                if n not in ignore_modules]
        self.children = OrderedDict()
        self.has_weight = hasattr(module, 'weight')
        self.has_bias = hasattr(module, 'bias')
        self.has_param = (self.has_weight or self.has_bias)
        self.has_param = (
                self.has_param and
                (self.active_only == '' or self.active_only == self.name))
        self.Ai0 = torch.Tensor(0)
        self.Ais = torch.Tensor(0)
        self.Gos = torch.Tensor(0)
        # self.Go = torch.Tensor(0)
        self._register_hooks()

    def _register_hooks(self):
        if not self.has_param:
            return
        logging.info(
                'Gluster> register hooks {} {}'.format(self.name, self.module))
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        self.module.register_forward_pre_hook(self._save_input_hook)
        self.module.register_backward_hook(self._save_dist_hook)

    def _save_input_hook(self, m, Ai):
        if not self.is_active:
            return
        if not self.has_param:
            return
        Ai0 = Ai[0]
        # Ai0 = self._post_proc_Ai(Ai0)
        if self.Ai0.shape != Ai0.shape:
            self.Ai0 = torch.zeros_like(Ai0).cuda()
        # self.Ai.copy_(Ai0)
        self.Ai0.copy_(Ai0)
        # del Ai0
        # if self.Ai.max() > 100000:
        #     # TODO: seed 1 this happens, try toy tests
        #     import ipdb; ipdb.set_trace()

    def _post_proc_Ai(self, Ai0):
        return Ai0

    def _post_proc_Go(self, Go0):
        return Go0

    def _save_dist_hook(self, m, grad_input, grad_output):
        if not self.is_active:
            return
        if not self.has_param:
            return
        with torch.no_grad():
            # Ai = self.Ai
            # Gi = grad_input[0].clone().detach()  # has 3 elements, 1 is Gi?
            Go0 = grad_output[0]
            # Go0 = self._post_proc_Go(Go0)
            # if self.Go0.shape != Go0.shape:
            #     self.Go0 = torch.zeros_like(Go0).cuda()
            # self.Go.copy_(Go0)
            Ai, Go = self._post_proc(Go0)
            # del Go0
            # Go = self.Go
            if self.no_grad:
                Go.fill_(1.)  # /Go0.numel())
            # print('%s %s' % (Ai.shape, Ai.shape))
            O = []
            # s = 0
            if self.module.bias is not None:
                O += [self._save_dist_hook_bias(Ai, Go)]
                # s += O[-1].sum()
            if self.module.weight is not None:
                O += [self._save_dist_hook_weight(Ai, Go)]
                # s += O[-1].sum()
            # if s < -10000:
            #     # TODO: seed 1 this happens, try toy tests
            #     import ipdb; ipdb.set_trace()
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
        if self.has_weight:
            self.Ci_new.fill_(0)
            self.Co_new.fill_(0)
        if self.has_bias:
            self.Cb_new.fill_(0)

    def update_batch(self, cluster_size):
        if not self.has_param:
            return
        if self.has_weight:
            self.Ci.copy_(self.Ci_new).div_(cluster_size.clamp(1))
            self.Co.copy_(self.Co_new).div_(cluster_size.clamp(1))
        if self.has_bias:
            self.Cb.copy_(self.Cb_new).div_(cluster_size.clamp(1))

    def update_online(self, pre_size, new_size):
        if not self.has_param:
            return
        # TODO: .4/5.5s overhead
        if self.has_weight:
            self.Ci.mul_(pre_size).add_(self.Ci_new).div_(new_size)
            self.Co.mul_(pre_size).add_(self.Co_new).div_(new_size)
        if self.has_bias:
            self.Cb.mul_(pre_size).add_(self.Cb_new).div_(new_size)

    def accum_new_centers(self, assign_i):
        if not self.has_param:
            return
        # TODO: SVD
        Ais = self.Ais
        Gos = self.Gos
        # TODO: .7/5.5s overhead
        if self.has_weight:
            self.Ci_new.scatter_add_(0, assign_i.expand_as(Ais), Ais)
            self.Co_new.scatter_add_(0, assign_i.expand_as(Gos), Gos)
        if self.has_bias:
            self.Cb_new.scatter_add_(0, assign_i.expand_as(Gos), Gos)

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
        raise Exception('Not implemented.')


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
        # logging.info('Gluster> container {}'.format(self.module))
        self._init_children(*args, **kwargs)
        self._set_default_loop()

    def _register_hooks(self):
        pass

    def _init_children(self, *args, **kwargs):
        for name, m in self.active_modules:
            glayer = get_gluster_module(m)
            if glayer is not None:
                kwargs['name'] = (
                        self.name+'.'
                        if self.name != '' else '')+name
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
        with torch.no_grad():
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
        # TODO: save dists 1.5/5.5s overhead
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
        if self.has_weight:
            self.Ci.masked_scatter_(reinits, self.Ais[perm])
            self.Co.masked_scatter_(reinits, self.Gos[perm])
        if self.has_bias:
            self.Cb.masked_scatter_(reinits, self.Gos[perm])

    def _post_proc(self, Go0):
        self.Ais = self.Ai0
        self.Gos = Go0
        return self.Ai0, Go0

    def get_centers(self):
        if not self.has_param:
            return
        centers = []
        if self.has_weight:
            Cf = torch.matmul(self.Co.unsqueeze(-1), self.Ci.unsqueeze(1))
            centers += [Cf.cpu().detach()]
            assert Cf.shape == (
                    self.Ci.shape[0], self.Co.shape[1], self.Ci.shape[1]
                    ), 'Cf: C x d_out x d_in'
        if self.has_bias:
            centers += [self.Cb.cpu().detach()]
        return centers


class GlusterConv(GlusterModule):
    def __init__(self, *args, **kwargs):
        # TODO: Roger, Martens, A Kronecker-factored approximate Fisher matrix
        # for convolution layer
        super(GlusterConv, self).__init__(*args, **kwargs)
        self.param = self.module.weight
        din = np.prod(list(self.param.shape)[1:])
        dout = self.param.shape[0]
        eps = self.eps
        C = self.nclusters
        with torch.no_grad():
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
        T = Go.shape[-1]

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
        # TODO: mean/sum?
        CG = torch.einsum('ko,bot->kb', [Cb, Go])/T
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
        # Ci = Ci.reshape((C, -1))
        # TODO: multi-GPU
        self.cost1 = cost1 = C*din*B*T + C*dout*B*T + C*B*T
        self.cost2 = cost2 = B*din*T*dout + C*B*din*dout
        if self.batch_dist is None:
            logging.info(
                    'Cost ratio %s: %.4f'
                    % (self.name, 1.*self.cost1/self.cost2))
        if cost1 < cost2:
            CiAi = torch.einsum('ki,bit->kbt', [Ci, Ai])
            CoGo = torch.einsum('ko,bot->kbt', [Co, Go])
            # worse in memory, maybe because of its internal cache
            # batch matmul matchs from the end of tensor2 backwards
            # CiAi = torch.matmul(Ci.unsqueeze(1).unsqueeze(1), Ai).squeeze(2)
            # CoGo = torch.matmul(Co.unsqueeze(1).unsqueeze(1), Go).squeeze(2)
            # TODO: mean/sum?
            CG = torch.einsum('kbt,kbt->kb', [CiAi, CoGo])/T
        else:
            AiGo = torch.einsum('bit,bot->bio', [Ai, Go])
            # TODO: mean/sum?
            CG = torch.einsum('ki,bio,ko->kb', [Ci, AiGo, Co])/T
        assert CG.shape == (C, B), 'CG: C x B.'
        CiCi = (Ci*Ci).view(Ci.shape[0], -1).sum(1)
        CoCo = (Co*Co).view(Co.shape[0], -1).sum(1)
        CC = ((CiCi)*(CoCo)).unsqueeze(-1)
        assert CC.shape == (C, 1), 'CC: C x 1.'
        O = CC-2*CG
        assert O.shape == (C, B), 'O: C x B.'
        # if self.Ai.max() > 10000:
        #     # TODO: seed 1 this happens, try toy tests
        #     import ipdb; ipdb.set_trace()

        return O

    def reinit_from_data(self, reinits, perm):
        if not self.has_param:
            return
        # TODO: mean/sum?
        # T = self.Go.shape[-1]
        # Ai = self.Ai[perm].sum(-1)/T
        # Go = self.Go[perm].sum(-1)/T
        Ais = self.Ais
        Gos = self.Gos
        if self.has_weight:
            self.Ci.masked_scatter_(reinits, Ais)
            self.Co.masked_scatter_(reinits, Gos)
        if self.has_bias:
            self.Cb.masked_scatter_(reinits, Gos)

    def _post_proc(self, Go0):
        module = self.module
        Ai = F.unfold(
                self.Ai0, module.kernel_size,
                padding=module.padding,
                stride=module.stride, dilation=module.dilation)
        Go = Go0.reshape(Go0.shape[:-2]+(np.prod(Go0.shape[-2:]), ))
        # TODO: mean/sum?
        T = Go.shape[-1]
        self.Ais = Ai.sum(-1)/T
        self.Gos = Go.sum(-1)/T
        return Ai, Go

    def get_centers(self):
        if not self.has_param:
            return
        C = self.nclusters
        dout = self.module.weight.shape[0]
        din = self.module.weight.shape[1:]
        centers = []
        # TODO: dist to centers is bad, maybe because of /T and numerical prec
        if self.has_weight:
            Cf = torch.matmul(self.Co.unsqueeze(-1), self.Ci.unsqueeze(1))
            assert Cf.shape == (C, dout, np.prod(din)), 'Cf: C x din x dout'
            Cf = Cf.reshape((C, ) + self.module.weight.shape)
            assert Cf.shape == (C, dout, )+din, 'Cf: C x co x ci x H x W'
            centers += [Cf.cpu().detach()]
        if self.has_bias:
            centers += [self.Cb.cpu().detach()]
        return centers
