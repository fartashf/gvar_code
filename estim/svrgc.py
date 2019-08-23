import torch

from .svrg import SVRGEstimator


class SVRGClipEstimator(SVRGEstimator):
    def __init__(self, *args, **kwargs):
        super(SVRGClipEstimator, self).__init__(*args, **kwargs)
        self.state = {}

    def grad(self, model_new, in_place=False):
        data = next(self.data_iter)

        model_old = self.model

        # old grad
        loss = model_old.criterion(model_old, data)
        g_old = torch.autograd.grad(loss, model_old.parameters())

        # new grad
        loss = model_new.criterion(model_new, data)
        g_new = torch.autograd.grad(loss, model_new.parameters())

        self.clip_r = 0
        self.clip_c = 0
        self.mn = 1e7
        self.mx = -1e-7
        if in_place:
            for m, go, gn, p in zip(
                    self.mu, g_old, g_new, model_new.parameters()):
                p.grad.copy_(self.c_clip(m-go, p)+gn)
            return loss
        ge = [self.c_clip(m-go, p)+gn
              for m, go, gn, p
              in zip(self.mu, g_old, g_new, model_new.parameters())]
        print('%.2f\t %.7f\t %.7f' % ((self.clip_r*100/self.clip_c).item(),
                                      self.mn, self.mx))
        return ge

    def c_clip(self, v, p):
        eps = 1e-7
        ftype = 2
        if ftype == 1:
            # adam stats
            state = self.optimizer.state[p]
        elif ftype == 2:
            # self stats
            state = self.state[p]
        E_x = state['exp_avg']
        E_s2, E_x2 = state['exp_avg_s2'], state['exp_avg_sq']

        ftype = 3
        if ftype == 1:
            # SNR over time
            # can be negative
            # 0, if Ps=Pnoise
            # denom = E_x2 - E_x*E_x
            snr = (E_x2+eps).log()-(E_s2+eps).log()
            clip = (snr > self.opt.svrgc_clip).float()
            self.mn = min(self.mn, snr.min().item())
            self.mx = max(self.mx, snr.max().item())
        elif ftype == 2:
            # norm-var
            nv = E_s2/(E_x2+eps)
            clip = (nv > self.opt.svrgc_clip).float()
            self.mn = min(self.mn, nv.min().item())
            self.mx = max(self.mx, nv.max().item())
        elif ftype == 3:
            # sign flip
            clip = (E_x.abs() > self.opt.svrgc_clip).float()
            self.mn = min(self.mn, E_x.abs().min().item())
            self.mx = max(self.mx, E_x.abs().max().item())
        v_clip = clip * v
        self.clip_r += clip.sum()
        self.clip_c += clip.numel()
        return v_clip
        # return 0

    def snap_online(self, model, niters):
        data = next(self.data_iter)
        loss = model.criterion(model, data)
        grad = torch.autograd.grad(loss, model.parameters())
        self.update_stats(model.parameters(), grad)

    def update_stats(self, param, grad):
        # beta1, beta2 = .9, .999
        beta1, beta2 = .9, .9
        for p, g in zip(param, grad):
            if p not in self.state:
                self.state[p] = {}
                self.state[p]['exp_avg'] = torch.zeros_like(p)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p)
                self.state[p]['exp_avg_s2'] = torch.zeros_like(p)
            state = self.state[p]
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg_s2 = state['exp_avg_s2']
            exp_avg.mul_(beta1).add_(1 - beta1, g)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, g, g)
            exp_avg_s2.mul_(beta2).addcmul_(1 - beta2, g-exp_avg, g-exp_avg)
