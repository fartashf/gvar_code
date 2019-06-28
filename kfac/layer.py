import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import logging


def get_module(module):
    module_name = module.__class__.__name__
    glayers = {'Linear': Linear, 'Conv2d': Conv2d}
    if module_name in glayers:
        return glayers[module_name]
    if len(list(module.children())) > 0:
        return Container
    return None


class Module(object):
    def __init__(
            self, module, no_grad=False, no_act=False,
            inactive_mods=[], active_only=[], name='',
            debug=True, stable=100, *args, **kwargs):
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
        self.stable = stable
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
            self.AtA, self.GtG = self._compute_cov(Ai, Go)

    def get_fisher(self):
        raise Exception("do not use")
        if not self.has_param:
            return
        # called after a backward
        return self.batch_fisher

    def activate(self):
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def update_inv(self):
        eps = 1e-10

        self.d_a, self.Q_a = torch.symeig(self.AtA, eigenvectors=True)
        self.d_g, self.Q_g = torch.symeig(self.GtG, eigenvectors=True)

        self.d_a.mul_((self.d_a > eps).float())
        self.d_g.mul_((self.d_g > eps).float())

    def get_natural_grad(self, grad, damping):
        p_grad_mat = self._get_matrix_form_grad(grad)

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

        nat_grad = self._get_grad_from_matrix(v, grad)
        return nat_grad

    def _get_matrix_form_grad(self, grad):
        raise NotImplementedError()

    def _get_grad_from_matrix(self, v, grad):
        raise NotImplementedError()

    def get_precond_eigs(self):
        if not self.has_param:
            return
        with torch.no_grad():
            d_a = torch.symeig(self.AtA)[0]
            d_g = torch.symeig(self.GtG)[0]
            evals = torch.einsum('i,j->ij', d_g, d_a).flatten()
        return evals


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
            glayer = get_module(m)
            if glayer is not None:
                kwargs['name'] = (
                        self.name+'.'
                        if self.name != '' else '')+name
                self.children[m] = glayer(*((m, )+args[1:]), **kwargs)

    def _set_default_loop(self):
        loop_functions = ['deactivate', 'activate',
                          'update_inv']
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
        return Ai, Go

    def _compute_cov(self, Ai, Go):
        B = Ai.shape[0]
        din = Ai.shape[1]
        dout = Go.shape[1]

        if self.has_bias:
            Ai = torch.cat([Ai, Ai.new(B, 1).fill_(1)], 1)
        AtA = Ai.t() @ (Ai / B)
        GtG = Go.t() @ (Go * B)
        assert AtA.shape == (din, din), 'AtA: din x din'
        assert GtG.shape == (dout, dout), 'GtG: dout x dout'
        return AtA, GtG

    def _get_matrix_form_grad(self, grad):
        # TODO: inplace grad vs outplace
        if self.has_bias:
            p_grad_mat = torch.cat([p_grad_mat])

    def _get_grad_from_matrix(self, v, grad):
        pass


class Conv2d(Module):
    def __init__(self, *args, **kwargs):
        # TODO: Roger, Martens, A Kronecker-factored approximate Fisher matrix
        # for convolution layer
        super(Conv2d, self).__init__(*args, **kwargs)
        self._conv_dot = None

    def _save_fisher_hook(self, Ai, Go):
        B = Ai.shape[0]
        T = Go.shape[-1]
        din, dout = self.din, self.dout

        assert Go.shape == (B, dout, T), 'Go: B x dout x T'
        assert Ai.shape == (B, din, T), 'Ai: B x din x T'
        # TODO: multi-GPU
        if self._conv_dot is None:
            self._conv_dot = self._conv_dot_choose(Ai, Go)
        GoG = self._conv_dot(Ai, Go)
        assert GoG.shape == (B, B), 'GoG: B x B.'

        return GoG

    def _conv_dot_choose(self, Ai, Go):
        din, dout = self.din, self.dout
        B = Ai.shape[0]
        T = Go.shape[-1]
        self.cost1 = cost1 = B*din*B*T*T + B*dout*B*T*T + B*B*T
        self.cost2 = cost2 = B*din*T*dout + B*B*din*dout
        if self.batch_fisher is None and self.debug:
            logging.info(
                    'GoG Cost ratio %s: %.4f'
                    % (self.name, 1.*self.cost1/self.cost2))
        if cost1 < cost2:
            return self._conv_dot_type1
        return self._conv_dot_type2

    def _conv_dot_type1(self, Ai, Go):
        raise Exception('Do not use')
        T = Go.shape[-1]
        B = Go.shape[0]

        # TODO: can we get rid of s?
        AiAi = torch.einsum('kit,bis->kbts', [Ai/T, Ai/B/T])
        GoGo = torch.einsum('kot,bos->kbts', [Go*T, Go*B*T])
        # worse in memory, maybe because of its internal cache
        # batch matmul matchs from the end of tensor2 backwards
        # CiAi = torch.matmul(Ci.unsqueeze(1).unsqueeze(1), Ai).squeeze(2)
        # CoGo = torch.matmul(Co.unsqueeze(1).unsqueeze(1), Go).squeeze(2)
        # mean/sum? sum
        GoG = torch.einsum('kbts,kbts->kb', [AiAi, GoGo])  # /T/T
        return GoG

    def _conv_dot_type2(self, Ai, Go):
        T = Go.shape[-1]
        # B = Go.shape[0]

        AiGo = torch.einsum('bit,bot->bio', [Ai/T, Go*T])
        # mean/sum? sum
        GoG = torch.einsum('kio,bio->kb', [AiGo, AiGo])  # /T
        return GoG

    def _conv_dot_type3(self, Ai, Go):
        raise Exception('Do not use')
        AiAi = torch.einsum('kis,bis->kb', [Ai, Ai])
        GoGo = torch.einsum('kos,bos->kb', [Go, Go])
        GoG = (AiAi) * (GoGo)
        return GoG

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
        return Ai, Go
