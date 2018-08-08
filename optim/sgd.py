from __future__ import print_function
import torch
import torch.optim as optim


class SimpleSGD(optim.Optimizer):
    def __init__(self, params, lr=0, momentum=0, **kwargs):
        defaults = dict(lr=lr, momentum=momentum)
        super(SimpleSGD, self).__init__(params, defaults)

    def step(self, **kwargs):
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                param_state = self.state[p]
                d_p = p.grad.data
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p)
                d_p = buf
                p.data.add_(-group['lr'], d_p)


def optim_log(optimizer, logger):
    gv = 0
    gn = 0
    pn = 0
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if momentum != 0:
                param_state = optimizer.state[p]
                if 'momentum_buffer' in param_state:
                    buf = param_state['momentum_buffer']
                    gv += (buf-(d_p+weight_decay*p.data)).pow(2).sum()
                    gn += buf.pow(2).sum()
                    pn += torch.numel(buf)

    logger.update('grad_var', gv/(pn + 1e-7), 1)
    logger.update('grad_var_n', gv/(gn + 1e-7), 1)
    logger.update('gbar_norm', gn, 1)


def sma_update(optimizer, sma_momentum):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.data is None:
                continue
            param_state = optimizer.state[p]
            if 'sma_buffer' not in param_state:
                buf = param_state['sma_buffer'] =\
                    torch.zeros_like(p.data)
            else:
                buf = param_state['sma_buffer']
            buf.mul_(sma_momentum).add_(p.data)
