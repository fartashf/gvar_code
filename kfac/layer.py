import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import logging
from cusvd import svdj


def get_module(module, no_indep=False):
    module_name = module.__class__.__name__
    if not no_indep:
        glayers = {'Linear': Linear, 'Conv2d': Conv2d}
    else:
        glayers = {'Linear': LinearNoIndep, 'Conv2d': Conv2d}
    if module_name in glayers:
        return glayers[module_name]
    if len(list(module.children())) > 0:
        return Container
    return None


class Module(object):
    def __init__(
            self, module, damping, no_grad=False, no_act=False,
            inactive_mods=[], active_only=[], name='',
            debug=True, *args, **kwargs):
        self.module = module
        self.is_active = False
        self.no_grad = no_grad  # grad=1 => clustring in the input space
        self.no_act = no_act  # act=1 => clustring in the grad wrt act space
        self.inactive_mods = inactive_mods
        self.name = name
        # TODO: unnamed children
        self.active_modules = [(n, m) for n, m in module.named_children()]
        self.children = OrderedDict()
        self.has_weight = hasattr(module, 'weight')
        self.has_bias = hasattr(module, 'bias')
        self.has_param = (self.has_weight or self.has_bias)
        self.has_param = (
                self.has_param and
                # (len(active_only) == 0 or self.name in active_only)
                (len(active_only) == 0 or
                    any(s in self.name for s in active_only))
                and
                # (len(inactive_mods) == 0 or self.name not in inactive_mods))
                (len(inactive_mods) == 0 or
                    not any(s in self.name for s in inactive_mods)))
        self.Ai0 = torch.Tensor(0)
        self.debug = debug
        self.damping = damping
        self._register_hooks()

        # covariance matrices
        self.AtA = None
        self.GtG = None

        # eigen decomposition
        self.d_a, self.Q_a = None, None
        self.d_g, self.Q_g = None, None

    def _register_hooks(self):
        if not self.has_param:
            return
        param = self.module.weight
        self.din = np.prod(param.shape[1:])
        self.dout = param.shape[0]

        if self.debug:
            logging.info(
                    '> register hooks {} {}'.format(
                        self.name, self.module))
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        self.module.register_forward_pre_hook(self._save_input_hook)
        self.module.register_backward_hook(self._save_grad_hook)

    def _save_input_hook(self, m, Ai):
        if not self.is_active:
            return
        if not self.has_param:
            return
        Ai0 = Ai[0]
        if self.Ai0.shape != Ai0.shape:
            self.Ai0 = torch.zeros_like(Ai0).cuda()
        self.Ai0.copy_(Ai0)

    def _save_grad_hook(self, m, grad_input, grad_output):
        if not self.is_active:
            return
        if not self.has_param:
            return
        with torch.no_grad():
            # Ai = self.Ai
            # Gi = grad_input[0].clone().detach()  # has 3 elements, 1 is Gi?
            Go0 = grad_output[0]
            Ai, Go = self._post_proc(Go0)
            # TODO: moving average
            self.Ai, self.Go = Ai, Go
            self.AtA, self.GtG = self._compute_cov(Ai, Go)

    def activate(self):
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def update_inv(self):
        if not self.has_param:
            return
        eps = 1e-10

        self.d_a, self.Q_a = torch.symeig(self.AtA, eigenvectors=True)
        self.d_g, self.Q_g = torch.symeig(self.GtG, eigenvectors=True)

        self.d_a.mul_((self.d_a > eps).float())
        self.d_g.mul_((self.d_g > eps).float())

    def precond_inplace(self):
        if not self.has_param:
            return
        p_grad_mat = self._get_matrix_from_grad()
        nat_grad = self._get_natural_grad(p_grad_mat)
        self._set_grad_from_matrix(nat_grad)

    def precond_outplace(self, g):
        raise NotImplementedError("recursive arguments")

    def _get_natural_grad(self, p_grad_mat):
        damping = self.damping
        v1 = self.Q_g.t() @ p_grad_mat @ self.Q_a
        v2 = v1 / (self.d_g.unsqueeze(1) *
                   self.d_a.unsqueeze(0) + damping)
        v = self.Q_g @ v2 @ self.Q_a.t()
        if self.has_bias:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(self.module.weight.size())
            v[1] = v[1].view(self.module.bias.size())
        else:
            v = [v.view(self.module.weight.size())]
        return v

    def _get_matrix_from_grad(self):
        raise NotImplementedError("Implemented by sub-classes.")

    def _set_grad_from_matrix(self, v):
        self.module.weight.grad.copy_(v[0])
        if self.has_bias:
            self.module.bias.grad.copy_(v[1])

    def get_precond_eigs(self):
        if not self.has_param:
            return
        d_a = torch.symeig(self.AtA)[0]
        d_g = torch.symeig(self.GtG)[0]
        evals = torch.einsum('i,o->io', d_g, d_a).flatten()
        return [evals]


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


class Container(Module):
    def __init__(self, *args, **kwargs):
        super(Container, self).__init__(*args, **kwargs)
        self._init_children(*args, **kwargs)
        self._set_default_loop()

    def _register_hooks(self):
        pass

    def _init_children(self, *args, **kwargs):
        for name, m in self.active_modules:
            glayer = get_module(m, no_indep=kwargs['no_indep'])
            if glayer is not None:
                kwargs['name'] = (
                        self.name+'.'
                        if self.name != '' else '')+name
                self.children[m] = glayer(*((m, )+args[1:]), **kwargs)

    def _set_default_loop(self):
        loop_functions = ['deactivate', 'activate',
                          'update_inv', 'precond_inplace', 'get_precond_eigs',
                          'precond_outplace']
        for fname in loop_functions:
            setattr(self, fname, LoopFunction(self, self.children, fname))


class Linear(Module):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)

    def _post_proc(self, Go0):
        if self.no_grad:
            Go0.fill_(1e-3)  # TODO: /Go0.numel())
        if self.no_act:
            self.Ai0.fill_(1e-3)  # TODO: /Go0.numel())
        Ai, Go = self.Ai0, Go0
        B = Ai.shape[0]
        if self.has_bias:
            Ai = torch.cat([Ai, Ai.new(B, 1).fill_(1)], 1)
        return Ai, Go

    def _compute_cov(self, Ai, Go):
        B = Ai.shape[0]
        din = Ai.shape[1]
        dout = Go.shape[1]

        AtA = Ai.t() @ (Ai / B)  # E[AtA], /B
        GtG = Go.t() @ (Go * B)  # E[GtG], xB because of batch averaged loss
        assert AtA.shape == (din, din), 'AtA: din x din'
        assert GtG.shape == (dout, dout), 'GtG: dout x dout'
        return AtA, GtG

    def _get_matrix_from_grad(self):
        p_grad_mat = self.module.weight.grad
        if self.has_bias:
            p_grad_mat = torch.cat(
                [p_grad_mat, self.module.bias.grad.view(-1, 1)], 1)
        return p_grad_mat


class LinearNoIndep(Linear):
    def __init__(self, *args, **kwargs):
        super(LinearNoIndep, self).__init__(*args, **kwargs)

    def update_inv(self):
        return

    def _get_natural_grad(self, p_grad_mat):
        eps = 1e-10
        damping = self.damping

        B = self.Ai.shape[0]
        AtG = torch.einsum('bi,bo->boi', self.Ai/B, self.Go*B)
        AtG = AtG.view(B, -1).contiguous()
        U, S, V = svdj(AtG)
        # U, S, V = torch.svd(AtG)
        S.mul_((S > eps).float())
        Si = S*S*B+damping
        # Si = 1
        v = V @ ((V.t() @ AtG.sum(0)) / Si)
        din = self.Ai.shape[1]
        v = v.view(self.dout, din)

        if self.has_bias:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(self.module.weight.size())
            v[1] = v[1].view(self.module.bias.size())
        else:
            v = [v.view(self.module.weight.size())]
        return v

    def get_precond_eigs(self):
        if not self.has_param:
            return
        with torch.no_grad():
            ftype = 1
            if ftype == 1:
                B = self.Ai.shape[0]
                AtG = torch.einsum('bi,bo->bio', self.Ai/B, self.Go*B)
                AtG = AtG.view(B, -1).contiguous()
                S = svdj(AtG)[1].flatten()
                evals = S*S*B  # AtG = d(1/B sum l_i) / d W
            elif ftype == 2:
                d_a = torch.symeig(self.Ai.t() @ self.Ai)[0]
                d_g = torch.symeig(self.Go.t() @ self.Go)[0]
                evals = torch.einsum('i,o->io', d_g, d_a).flatten()
        return [evals]


class Conv2d(Module):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)

    def _post_proc(self, Go0):
        if self.no_grad:
            Go0.fill_(1e-3)  # TODO: /Go0.numel())
        if self.no_act:
            self.Ai0.fill_(1e-3)  # TODO: /Go0.numel())
        module = self.module
        Ai = F.unfold(
                self.Ai0, module.kernel_size,
                padding=module.padding,
                stride=module.stride, dilation=module.dilation)
        Go = Go0.reshape(Go0.shape[:-2]+(np.prod(Go0.shape[-2:]), ))
        B = Ai.shape[0]
        T = Ai.shape[-1]
        if self.has_bias:
            Ai = torch.cat([Ai, Ai.new(B, 1, T).fill_(1)], 1)
        return Ai, Go

    def _compute_cov(self, Ai, Go):
        B = Ai.shape[0]
        din = Ai.shape[1]
        dout = Go.shape[1]
        T = Go.shape[-1]

        assert Ai.shape == (B, din, T), 'Ai: B x din x T'
        assert Go.shape == (B, dout, T), 'Go: B x dout x T'

        # E[AtA], /B
        # E[GtG], xB because of batch averaged loss
        # in the KFC paper, it makes further assumptions
        # that those terms are spatially homogenuous..
        # so they have the same mean and variance.
        # Eq 32 and 74 in https://arxiv.org/pdf/1602.01407.pdf
        AtA = torch.einsum('bit,bot->io', Ai/T, Ai/T/B)
        GtG = torch.einsum('bot,bpt->op', Go*T, Go*T*B/T)  # an extra /T

        assert AtA.shape == (din, din), 'AtA: din x din'
        assert GtG.shape == (dout, dout), 'GtG: dout x dout'
        return AtA, GtG

    def _get_matrix_from_grad(self):
        p_grad_mat = self.module.weight.grad.view(self.dout, self.din)
        if self.has_bias:
            p_grad_mat = torch.cat(
                [p_grad_mat, self.module.bias.grad.view(-1, 1)], 1)
        return p_grad_mat


class Conv2dNoIndep(Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dNoIndep, self).__init__(*args, **kwargs)

    def get_precond_eigs(self):
        if not self.has_param:
            return
        with torch.no_grad():
            raise Exception('Check if spatial assumption is satisfied.')
            B = self.Ai.shape[0]
            AtG = torch.einsum('bit,bot->bio', self.Ai/B, self.Go*B)
            AtG = AtG.view(B, -1).contiguous()
            S = svdj(AtG)[1].flatten()
            evals = S*S*B  # AtG = d(1/B sum l_i) / d W
        return [evals]
