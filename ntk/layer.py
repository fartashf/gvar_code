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
            self, module, eps, no_grad=False, no_act=False,
            inactive_mods=[], active_only=[], name='',
            debug=True, stable=100, *args, **kwargs):
        self.module = module
        self.eps = eps
        self.batch_dist = None
        self.is_active = True
        self.is_eval = False
        self.no_grad = no_grad  # grad=1 => clustring in the input space
        self.no_act = no_act  # act=1 => clustring in the grad wrt act space
        self.inactive_mods = inactive_mods
        self.name = name
        # TODO: unnamed children
        self.active_modules = [(n, m) for n, m in module.named_children()]
        self.children = OrderedDict()
        self.has_weight = hasattr(module, 'weight')
        self.has_bias = False  # TODO: hasattr(module, 'bias') # SVD
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

    def _register_hooks(self):
        if not self.has_param:
            return
        if self.debug:
            logging.info(
                    '> register hooks {} {}'.format(
                        self.name, self.module))
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        self.module.register_forward_pre_hook(self._save_input_hook)
        self.module.register_backward_hook(self._save_dist_hook)

    def _save_input_hook(self, m, Ai):
        if not self.is_active:
            return
        if not self.has_param:
            return
        Ai0 = Ai[0]
        if self.Ai0.shape != Ai0.shape:
            self.Ai0 = torch.zeros_like(Ai0).cuda()
        self.Ai0.copy_(Ai0)

    def _save_dist_hook(self, m, grad_input, grad_output):
        if not self.is_active:
            return
        if not self.has_param:
            return
        with torch.no_grad():
            # Ai = self.Ai
            # Gi = grad_input[0].clone().detach()  # has 3 elements, 1 is Gi?
            Go0 = grad_output[0]
            Ai, Go = self._post_proc(Go0)
            O = []
            if self.has_bias:
                O += [self._save_dist_hook_bias(Ai, Go)]
            if self.has_weight:
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
        # logging.info('> container {}'.format(self.module))
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
        loop_functions = ['get_dist', 'deactivate', 'activate', 'eval']
        for fname in loop_functions:
            setattr(self, fname, LoopFunction(self, self.children, fname))


class Linear(Module):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        weight = self.module.weight
        dout, din = weight.shape

    def _save_dist_hook_bias(self, Ai, Go):
        raise Exception('not implemented')

    def _save_dist_hook_weight(self, Ai, Go):
        B = Ai.shape[0]

        AiAi = torch.matmul(Ai, Ai.t())
        GoGo = torch.matmul(Go, Go.t())
        GoG = (AiAi)*(GoGo)
        assert GoG.shape == (B, B), 'GoG: B x B.'
        return GoG

    def _post_proc(self, Go0):
        if self.no_grad:
            Go0.fill_(1e-3)  # TODO: /Go0.numel())
        if self.no_act:
            self.Ai0.fill_(1e-3)  # TODO: /Go0.numel())
        Ai, Go = self.Ai0, Go0
        return Ai, Go


class Conv2d(Module):
    def __init__(self, *args, **kwargs):
        # TODO: Roger, Martens, A Kronecker-factored approximate Fisher matrix
        # for convolution layer
        super(Conv2d, self).__init__(*args, **kwargs)
        self.param = self.module.weight
        self._gnorm = None
        self._conv_dot = None

    def _save_dist_hook_bias(self, Ai, Go):
        raise Exception('not implemented')

    def _save_dist_hook_weight(self, Ai, Go):
        param = self.param
        B = Ai.shape[0]
        din = np.prod(param.shape[1:])
        dout = param.shape[0]
        T = Go.shape[-1]

        assert Go.shape == (B, dout, T), 'Go: B x dout x T'
        assert Ai.shape == (B, din, T), 'Ai: B x din x T'
        # TODO: multi-GPU
        if self._conv_dot is None:
            self._conv_dot = self._conv_dot_choose(Ai, Go)
        GoG = self._conv_dot(Ai, Go)
        assert GoG.shape == (B, B), 'GoG: G x B.'

        return GoG

    def _conv_dot_choose(self, Ai, Go):
        B, din = Ai.shape
        B, dout, T = Go.shape
        self.cost1 = cost1 = B*din*B*T*T + B*dout*B*T*T + B*B*T
        self.cost2 = cost2 = B*din*T*dout + B*B*din*dout
        if self.batch_dist is None and self.debug:
            logging.info(
                    'GoG Cost ratio %s: %.4f'
                    % (self.name, 1.*self.cost1/self.cost2))
        if cost1 < cost2:
            return self._conv_dot_type1
        return self._conv_dot_type2

    def _conv_dot_type1(self, Ai, Go):
        # TODO: can we get rid of s?
        AiAi = torch.einsum('kit,bis->kbts', [Ai, Ai])
        GoGo = torch.einsum('kot,bos->kbts', [Go, Go])
        # worse in memory, maybe because of its internal cache
        # batch matmul matchs from the end of tensor2 backwards
        # CiAi = torch.matmul(Ci.unsqueeze(1).unsqueeze(1), Ai).squeeze(2)
        # CoGo = torch.matmul(Co.unsqueeze(1).unsqueeze(1), Go).squeeze(2)
        # mean/sum? sum
        GoG = torch.einsum('kbts,kbts->kb', [AiAi, GoGo])  # /T
        return GoG

    def _conv_dot_type2(self, Ci, Ai, Co, Go):
        AiGo = torch.einsum('bit,bot->bio', [Ai, Go])
        # mean/sum? sum
        GoG = torch.einsum('kio,bio->kb', [AiGo, AiGo])  # /T
        return GoG

    def _gnorm_choose(self, Ai, Go):
        B, din, T = Ai.shape
        B, dout, T = Go.shape
        self.cost1 = cost1 = B*din*T*T + B*dout*T*T + B*T*T
        self.cost2 = cost2 = B*din*dout*T + B*din*dout
        if self.batch_dist is None and self.debug:
            logging.info(
                    'GG Cost ratio %s: %.4f'
                    % (self.name, 1.*self.cost1/self.cost2))
        if cost1 < cost2:
            return self._gnorm_type1
        return self._gnorm_type2

    def _gnorm_type1(self, Ai, Go):
        AiAi = torch.einsum('bis,bit->bst', [Ai, Ai])
        GoGo = torch.einsum('bos,bot->bst', [Go, Go])
        # GG = torch.einsum('bst,bst->b', [AiAi, GoGo])
        GG = (AiAi*GoGo).sum(-1).sum(-1)
        return GG

    def _gnorm_type2(self, Ai, Go):
        # return Ai[:, 0, 0]*0+1
        AiGo = torch.einsum('bit,bot->bio', [Ai, Go])
        # return AiGo[:, 0, 0]*0+1
        # TODO: why?!!!!
        # GG = torch.einsum('bio,bio->b', [AiGo, AiGo])
        GG = (AiGo*AiGo).sum(-1).sum(-1)
        return GG

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
