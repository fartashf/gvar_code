from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


class AddDMom(optim.SGD):
    def __init__(self, params, opt, train_size=0, dmom=0, *args, **kwargs):
        self.alpha = Variable(torch.zeros(train_size).cuda(),
                              requires_grad=True)
        self.alpha.grad = Variable(torch.zeros(train_size).cuda())
        # self.alpha = np.zeros(train_size)
        self.ploss = np.zeros(train_size)
        params = list(params)+[self.alpha]
        super(AddDMom, self).__init__(params, *args, **kwargs)
        self.dmom = dmom
        self.epoch = 0
        self.opt = opt

    def step(self, idx, loss_i):
        idx = idx.cuda()
        # alpha_i = Variable(torch.Tensor(self.alpha[idx]).cuda(),
        #                    requires_grad=True)
        alpha_i = Variable(self.alpha[idx].data, requires_grad=True)
        loss = (loss_i - self.dmom*alpha_i + 1).clamp(min=0)
        numz = (loss_i+1 == alpha_i).data.cpu().numpy().sum()
        loss += alpha_i
        self.ploss[idx] = loss.data.cpu().numpy()
        loss = loss.sum()/len(idx)

        loss.backward()
        ga = alpha_i.grad.data.cpu().numpy()
        self.alpha.grad[idx] = alpha_i.grad
        super(AddDMom, self).step()
        # self.alpha[idx] = alpha_i.clamp(min=0).data.cpu().numpy()
        self.alpha.data = self.alpha.data.clamp(min=0)

        self.logger.update('alpha', self.alpha.data.cpu().numpy(),
                           1, hist=True)
        self.logger.update('ploss', self.ploss, 1, hist=True)
        self.logger.update('numz', numz, len(idx))
        self.logger.update('ga', ga, 1, hist=True)
        return loss
