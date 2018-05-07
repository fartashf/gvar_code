from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
# import time


def loss_jvp(W, loss_ex, m):
    # for example i (d/dW loss_ex)^T m
    # W = model.parameters()
    v = Variable(torch.ones_like(loss_ex.data), requires_grad=True).cuda()
    # create_graph=True vs retain_graph=True the difference fixed in v0.4.0
    grad_params = torch.autograd.grad(loss_ex, W, v, create_graph=True)
    # toc = time.time()
    # m = [torch.ones_like(g) for g in grad_params]
    jvp = torch.autograd.grad(grad_params, v, m)[0]
    # print('%.2f' % (time.time()-toc))
    return jvp


class DMomSGDJVP(optim.Optimizer):
    # http://pytorch.org/docs/master/_modules/torch/optim/sgd.html
    def __init__(self, params, opt, train_size, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        params = list(params)
        self.W = params
        super(DMomSGDJVP, self).__init__(params, defaults)
        self.opt = opt
        self.alpha_mom = np.zeros(train_size)
        self.alpha_mom_bc = np.zeros(train_size)
        self.alpha = np.ones((train_size,))
        self.alpha_normed = np.ones((train_size,))
        self.loss = np.ones((train_size,))
        self.weights = np.ones((train_size,))
        self.epoch = 0
        self.alpha_normed_pre = None
        self.g_bar = None

    def step(self, idx, loss, target, **kwargs):
        self.loss[idx] = loss.data.cpu().numpy()
        target = target.data.cpu().numpy().copy()

        self.profiler.tic()
        alpha = self.compute_alpha(loss)
        self.alpha[idx] = alpha
        self.logger.update('alpha', alpha, len(idx))
        self.logger.update('alpha_sum', float(alpha.sum()), len(idx))
        self.logger.update('alpha_sq_sum',
                           float((alpha*alpha).sum()), len(idx))

        alpha_mom, alpha_bc = self.compute_dmom(alpha, idx)
        self.alpha_mom[idx] = alpha_mom
        self.alpha_mom_bc[idx] = alpha_bc
        self.logger.update('alpha_mom', alpha_mom, len(idx))

        # TODO: low/high clip threshold
        # numz += (alpha_mom < self.low_theta).sum()
        # bigg += (alpha_mom > self.high_theta).sum()
        self.profiler.toc('alpha')

        alpha_norm = self.normalize_alpha(alpha_bc, target)
        self.alpha_normed[idx] = alpha_norm
        self.profiler.toc('norm')

        self.zero_grad()
        weights = torch.Tensor(self.weights[idx]/len(idx)).cuda()
        loss.backward(weights)
        self.profiler.toc('backward')

        gv, gvn = self.step_sgd()
        self.logger.update('grad_var', gv, len(idx))
        self.logger.update('grad_var_n', gvn, len(idx))
        if 'F' in self.opt.alpha:
            self.compute_moments()
        self.profiler.toc('update')

    def step_sgd(self):
        gv = 0
        gn = 0
        pn = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] =\
                            torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        gv += (buf-d_p).pow(2).sum()
                        gn += buf.pow(2).sum()
                        pn += torch.numel(buf)
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)
        return gv/(pn + 1e-7), gv/(gn + 1e-7)

    def compute_alpha(self, loss):
        g_bar = self.get_gbar()
        if self.opt.alpha == 'loss':
            alpha = loss.data.cpu().numpy()
        elif self.opt.alpha == 'gtheta_abs':
            alpha = np.abs(loss_jvp(self.W, loss, g_bar).data.cpu().numpy())
        elif self.opt.alpha == 'one' or g_bar is None:
            alpha = np.ones((len(loss), ))
        elif 'gbar_abs' in self.opt.alpha:
            alpha = np.abs(loss_jvp(self.W, loss, g_bar).data.cpu().numpy())
        elif 'gbar_max0' in self.opt.alpha:
            alpha = np.maximum(
                loss_jvp(self.W, loss, g_bar).data.cpu().numpy(),
                0)
        elif self.opt.alpha == 'lgg':
            alpha1 = loss.data.cpu().numpy()
            alpha1 /= alpha1.sum()
            alpha2 = np.abs(loss_jvp(self.W, loss, g_bar).data.cpu().numpy())
            alpha2 /= alpha2.sum()
            alpha = alpha1 + alpha2
        return alpha

    def get_gbar(self):
        if (self.g_bar is not None
                or self.opt.alpha == 'loss'
                or self.opt.alpha == 'one'):
            return self.g_bar

        if 'theta' in self.opt.alpha:
            self.g_bar = [w.data for w in self.W]
            return self.g_bar
        elif 'F' in self.opt.alpha:
            m = [self.state[p]['fgbar']
                 for group in self.param_groups
                 for p in group['params']
                 if p.grad is not None]
        # elif 'Figbar' in self.opt.alpha:
        #     m = [1/(1e-8+self.state[p]['exp_avg_sq_bc'])
        #          for group in self.param_groups
        #          for p in group['params']
        #          if p.grad is not None]
        elif 'gbar' in self.opt.alpha:
            m = [self.state[p]['momentum_buffer']
                 for group in self.param_groups
                 for p in group['params']
                 if p.grad is not None]
        if len(m) != 0:
            self.g_bar = m

        return self.g_bar

    def compute_dmom(self, alpha, idx):
        dmom = self.opt.dmom
        # alpha_mom
        alpha_mom = self.alpha_mom[idx]*dmom + alpha*(1-dmom)

        # bias correction (after storing)
        alpha_bc = alpha_mom/(1-dmom**(self.epoch+1))

        return alpha_mom, alpha_bc

    def normalize_alpha(self, alpha_val, target):
        # normalize alphas to sum to 1
        if self.opt.alpha_norm == 'sum':
            alpha_val /= alpha_val.sum()
        elif self.opt.alpha_norm == 'exp':
            alpha_val -= alpha_val.max()
            alpha_val = np.exp(alpha_val/self.opt.norm_temp)
            alpha_val /= alpha_val.sum()
        elif self.opt.alpha_norm == 'sum_batch':
            alpha_val /= self.alpha.sum()
        elif self.opt.alpha_norm == 'exp_batch':
            alpha_val -= self.alpha.max()
            salpha = self.alpha - self.alpha.max()
            alpha_val = np.exp(alpha_val/self.opt.norm_temp)
            salpha = np.exp(salpha)
            alpha_val /= salpha.sum()
        elif self.opt.alpha_norm == 'sum_class':
            for i in range(target.max()+1):
                cl_idx = np.where(target == i)[0]
                alpha_val[cl_idx] /= alpha_val[cl_idx].sum()
        elif self.opt.alpha_norm == 'exp_class':
            for i in range(target.max()+1):
                cl_idx = np.where(target == i)[0]
                alpha_val[cl_idx] -= alpha_val[cl_idx].max()
                alpha_val[cl_idx] = np.exp(
                    alpha_val[cl_idx]/self.opt.norm_temp)
                alpha_val[cl_idx] /= alpha_val[cl_idx].sum()
        elif self.opt.alpha_norm == 'none':
            alpha_val /= len(alpha_val)
        return alpha_val

    def log_perc(self, prefix=''):
        self.logger.update(prefix+'dmom_h', self.alpha_mom, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'alpha_h', self.alpha, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'alpha_normed_h',
                           self.alpha_normed, 1, hist=True, log_scale=True)
        self.logger.update(prefix+'loss_h', self.loss, 1, hist=True,
                           log_scale=True)
        if self.alpha_normed_pre is not None:
            sc = len(self.alpha_normed)-1-np.argsort(self.alpha_normed)
            sp = len(self.alpha_normed)-1-np.argsort(self.alpha_normed_pre)
            sa = np.maximum(sc, sp)
            saf = sa[np.where(sa < sa.size/10)[0]]
            self.logger.update(prefix+'big_alpha_vs_pre_h', saf, 1, hist=True)
        self.alpha_normed_pre = self.alpha_normed

    def compute_moments(self):
        """
        From Adam
        """
        beta2 = 0.999

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if 'exp_avg' not in state:
                    state['step'] = 0
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['fgbar'] = torch.zeros_like(p.data)

                exp_avg_sq = state['exp_avg_sq']
                fgbar = state['fgbar']
                mom = state['momentum_buffer']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction2 = 1 - beta2 ** state['step']

                # exp_avg_sq_bc.copy_(exp_avg_sq).div_(bias_correction2)
                if 'Fgbar' in self.opt.alpha:
                    fgbar.zero_().addcmul_(1/bias_correction2, mom, exp_avg_sq)
                elif 'Figbar' in self.opt.alpha:
                    fgbar.zero_().addcdiv_(bias_correction2, mom, exp_avg_sq)
