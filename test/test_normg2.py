import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import sys
sys.path.append('../')
from models.mnist import MLP, MNISTNet  # NOQA


class NormgComputer(object):
    def __init__(self, model):
        self.layer_normg2 = {'Linear': self.linear_normg2,
                             'Conv2d': self.conv2d_normg2}
        self.known_modules = self.layer_normg2.keys()

        self.model = model
        self.modules = []
        self.inputs = {}
        self.normg2_data = {}

        self._register_hooks()

    def _register_hooks(self):
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_normg2)

    def _save_input(self, module, input):
        self.inputs[module] = input[0].clone().detach()

    def _save_normg2(self, module, grad_input, grad_output):
        a_in = self.inputs[module]
        d_out = grad_output[0].clone().detach()
        # print('%s %s' % (a_in.shape, d_out.shape))
        # TODO: bias
        self.layer_normg2[module.__class__.__name__](module, a_in, d_out)

    def linear_normg2(self, module, a_in, d_out):
        a2 = a_in.pow(2).sum(1)
        d2 = d_out.pow(2).sum(1)
        normg2 = a2*d2
        assert len(normg2.shape) == 1, 'One dimension only'
        assert normg2.shape[0] == a2.shape[0], 'First dimension is batch_size'
        # print(normg2.shape)
        self.normg2_data[module] = normg2

    def conv2d_normg2(self, module, a_in, d_out):
        # TODO: conv
        pass


def test_normg2():
    batch_size = 10  # 128
    x = Variable(torch.rand(batch_size, 1, 28, 28), requires_grad=True).cuda()
    t = Variable(torch.ones(batch_size).long()).cuda()

    # model = MNISTNet(dropout=False)
    model = MLP(dropout=False)
    model.cuda()
    # W = list(model.parameters())
    W = list([param
              for name, param in model.named_parameters()
              if 'bias' not in name])
    normg2 = np.zeros(batch_size)
    for i in range(batch_size):
        model.zero_grad()
        y = model(x[i:i+1])
        loss = F.nll_loss(y, t[i:i+1])/batch_size
        grad_params = torch.autograd.grad(loss, W)
        normg2[i] = np.sum([g.pow(2).sum().item() for g in grad_params])
    print(normg2)

    modelg = copy.deepcopy(model)
    ghash = NormgComputer(modelg)

    modelg.zero_grad()
    y = modelg(x)
    loss = F.nll_loss(y, t)
    loss.backward()
    normg2_data = torch.stack(ghash.normg2_data.values())
    normg2_fast = normg2_data.sum(0).cpu().numpy()
    print(normg2_fast)
    print('Diff: %.4f' % np.power(normg2-normg2_fast, 2).sum())


if __name__ == '__main__':
    test_normg2()
