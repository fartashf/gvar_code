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

