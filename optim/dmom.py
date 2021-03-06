from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np


class DMomSGD(optim.Optimizer):
    # http://pytorch.org/docs/master/_modules/torch/optim/sgd.html
    def __init__(self, params, opt, train_size=0, lr=0, momentum=0, dmom=0,
                 weight_decay=0, update_interval=1):
        defaults = dict(lr=lr, momentum=momentum, dmom=dmom,
                        weight_decay=weight_decay, wmomentum=opt.wmomentum)
        super(DMomSGD, self).__init__(params, defaults)
        self.data_momentum = np.zeros(train_size)
        self.grad_norm = np.ones((train_size,))
        self.fnorm = np.ones((train_size,))
        self.alpha = np.ones((train_size,))
        self.alpha_normed = np.ones((train_size,))
        self.alpha_normed_pre = None
        self.loss = np.ones((train_size,))
        self.epoch = 0
        self.low_theta = opt.low_theta
        self.high_theta = opt.high_theta
        self.update_interval = update_interval
        self.opt = opt
        self.weights = np.ones((train_size,))
        self.alpha_mom_bc = np.ones((train_size,))

    def step(self, idx, grads_group, loss, target, **kwargs):
        # import numpy as np
        # print(np.histogram(self.data_momentum.cpu().numpy()))
        target = target.data.cpu().numpy().copy()
        data_mom = self.data_momentum
        numz = 0
        bigg = 0
        numo = 0
        nz_sample = 0
        gvw_mean = 0
        gv_mean = 0
        gw_g = 0
        gvv = 0
        gn = 0
        pn = 0
        temperature = (self.epoch/self.opt.epochs)**self.opt.dmom_temp
        normg_ratio = np.zeros((len(idx),))
        dmom_ratio = np.zeros((len(idx),))
        newdmom_normg_ratio = np.zeros((len(idx),))
        normg_val = np.zeros((len(idx),))
        dmom_val = np.zeros((len(idx),))
        fnorm_val = np.zeros((len(idx),))
        alpha_val = np.zeros((len(idx),))
        gc_norm = np.zeros((len(idx),))
        for group, grads in zip(self.param_groups, grads_group):
            self.profiler.tic()
            param_state = [self.state[p] for p in group['params']]
            if 'momentum_buffer' in param_state[0]:
                buf = [p['momentum_buffer'].view(-1) for p in param_state]
            else:
                buf = [p.data.view(-1) for p in group['params']]
            gg = torch.cat(buf)
            # gg_norm = np.sqrt(gg.pow(2).sum())
            dmom = group['dmom']
            grad_acc = [torch.zeros_like(g.data).cuda() for g in grads[0]]
            grad_acc_w = [torch.zeros_like(g.data).cuda() for g in grads[0]]
            grad_var_w = [torch.zeros_like(g.data).cuda() for g in grads[0]]
            grad_var = [torch.zeros_like(g.data).cuda() for g in grads[0]]
            for i in range(len(idx)):
                gf = [g.data.view(-1) for g in grads[i]]
                gc = torch.cat(gf)
                gc_norm[i] = np.sqrt(gc.pow(2).sum())

                # current normg, dmom
                normg = torch.pow(gc, 2).sum(dim=0, keepdim=True).sqrt(
                ).cpu().numpy().copy()
                if self.opt.alpha == 'one':
                    alpha = 1
                elif self.opt.alpha == 'normg':
                    alpha = normg
                elif self.opt.alpha == 'ggbar_abs':
                    # normg = 1/(1+np.exp(abs(torch.dot(gc, gg))
                    #                     - self.opt.jacobian_lambda))
                    # normg = abs(torch.dot(gc, gg))
                    # from scipy.special import expit
                    # normg = expit(-abs(torch.dot(gc, gg))
                    #               * self.opt.jacobian_lambda + 100)
                    alpha = abs(torch.dot(gc, gg))
                elif self.opt.alpha == 'ggbar_max0':
                    alpha = max(0, torch.dot(gc, gg))
                elif self.opt.alpha == 'loss':
                    alpha = loss[i].data.cpu().numpy().copy()
                elif self.opt.alpha == 'normtheta':
                    alpha = torch.dot(gc, gg)
                    # alpha += torch.dot(gc, theta)
                    alpha = abs(alpha)
                elif self.opt.alpha == 'rand':
                    prob, aval = map(float, self.opt.sampler_params.split(','))
                    if np.random.randint(100) < prob*100:
                        alpha = aval
                    else:
                        alpha = 1

                alpha0 = alpha

                # alpha_mom
                new_dmom = data_mom[idx[i]]*dmom + alpha*(1-dmom)

                # bias correction (after storing)
                alpha = new_dmom/(1-dmom**(self.epoch+1))

                # temperature
                alpha = alpha**temperature

                # last normg, dmom
                last_dmom = float(data_mom[idx[i]:idx[i]+1])
                last_normg = float(self.grad_norm[idx[i]:idx[i]+1])
                fnorm = abs(torch.dot(gc, gg))

                # update normg, dmom
                if self.epoch % self.update_interval == 0:
                    data_mom[idx[i]:idx[i]+1] = new_dmom
                    self.grad_norm[idx[i]:idx[i]+1] = normg
                    self.fnorm[idx[i]:idx[i]+1] = fnorm
                    self.alpha[idx[i]:idx[i]+1] = alpha0
                    self.alpha_mom_bc[idx[i]:idx[i]+1] = alpha
                    self.loss[idx[i]:idx[i]+1] = float(loss[i])

                # dt = self.opt.dmom_theta
                # dr = new_dmom/(normg+self.low_theta)
                # if dt > 1 and dr > dt:
                #     numo += 1
                #     new_dmom = normg*dt

                # ratios
                normg_ratio[i] = (normg-last_normg)
                dmom_ratio[i] = (new_dmom-last_dmom)
                newdmom_normg_ratio[i] = (new_dmom/(normg+self.low_theta))
                normg_val[i] = normg
                dmom_val[i] = new_dmom
                fnorm_val[i] = fnorm
                alpha_val[i] = alpha

                if float(normg) < self.low_theta:
                    numz += 1
                if float(normg) > self.high_theta:
                    bigg += 1
                    continue
            self.profiler.toc('update')

            # normalize alphas to sum to 1
            if self.opt.alpha_norm == 'sum':
                alpha_val /= alpha_val.sum()
            elif self.opt.alpha_norm == 'exp':
                alpha_val -= alpha_val.max()
                alpha_val = np.exp(alpha_val/self.opt.norm_temp)
                alpha_val /= alpha_val.sum()
            elif self.opt.alpha_norm == 'normg':
                alpha_val /= normg_val[i]+self.low_theta
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
                # TODO: sum to one
                pass
            elif self.opt.alpha_norm == 'bs':
                alpha_val /= len(idx)
            self.profiler.toc('norm')
            self.alpha_normed[idx] = alpha_val

            # accumulate current mini-batch weighted grad
            sampler_alpha_th = self.opt.sampler_alpha_th
            if self.opt.sampler_alpha_perc > 0:
                sampler_alpha_th = float(
                    np.percentile(alpha_val, self.opt.sampler_alpha_perc))
            # TODO: divsum
            # ws = self.weights[idx].sum()
            if self.opt.instantw:
                lmbd = float(self.opt.sampler_params)
                self.weights[idx] = .5*lmbd/alpha_val.clip(1e-20)
                self.weights.clip(0, self.opt.sampler_maxw)
            ws = len(idx)
            for i in range(len(idx)):
                sa_perc = self.opt.sampler_alpha_perc
                if (alpha_val[i] < sampler_alpha_th
                        or (sa_perc > 0 and
                            i-nz_sample >= (100.-sa_perc)*len(idx)/100.)):
                    nz_sample += 1
                    continue
                for g, ga, gaw in zip(grads[i], grad_acc, grad_acc_w):
                    # ga += g.data*float(
                    #     data_mom[idx[i]:idx[i]+1]/(normg+self.low_theta))
                    if self.opt.sampler:
                        gaw += g.data/len(idx)
                        ga += g.data*self.weights[idx[i]]/ws  # no /len(idx)
                    else:
                        # AAA = g.data*alpha_val[i]
                        # if self.opt.divlen != 0:
                        #     AAA /= len(idx)
                        # gaw += AAA  # TODO: no /len
                        gaw += g.data*alpha_val[i]
                        ga += g.data/len(idx)
            # grad variance
            for i in range(len(idx)):
                sa_perc = self.opt.sampler_alpha_perc
                if (alpha_val[i] < sampler_alpha_th
                        or (sa_perc > 0 and
                            i-nz_sample >= (100.-sa_perc)*len(idx)/100.)):
                    continue
                for g, ga, gaw, gvw, gv in zip(grads[i], grad_acc, grad_acc_w,
                                               grad_var_w, grad_var):
                    if self.opt.sampler:
                        gvw += (g.data-gaw).pow(2)
                        gv += (g.data*self.weights[idx[i]]/ws-ga).pow(2)
                    else:
                        gvw += (g.data*alpha_val[i]-gaw).pow(2)
                        gv += (g.data-ga).pow(2)
            gvw_sum = sum([gvw_.sum() for gvw_ in grad_var_w])
            gv_sum = sum([gv_.sum() for gv_ in grad_var])
            if self.opt.sampler:
                gvw_sum /= len(idx)
                gv_sum /= len(idx)-nz_sample
            else:
                gv_sum /= len(idx)
                gvw_sum /= len(idx)-nz_sample
            param_num = sum([gvw_.numel() for gvw_ in grad_var_w])
            # from "Backpropagation through the Void":
            # the sample log-variance is reported
            # averaged over all policy parameters
            gvw_mean = gvw_sum/param_num
            gv_mean = gv_sum/param_num
            # import ipdb; ipdb.set_trace()
            self.profiler.toc('accum')

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            wmomentum = group['wmomentum']
            gw_norm = 0
            g_norm = 0
            for p, ga, gaw in zip(group['params'], grad_acc, grad_acc_w):
                param_state = self.state[p]
                # TODO: more numerical precision
                # d_p = ga/len(idx)
                # d_p_w = gaw/len(idx)
                d_p = ga
                d_p_w = gaw
                gw_g += (d_p-d_p_w).pow(2).sum()
                gw_norm += d_p_w.pow(2).sum()
                g_norm += d_p.pow(2).sum()
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    d_p_w.add_(weight_decay, p.data)
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                    buf_w = param_state['wmomentum_buffer'] = torch.zeros_like(
                        p.data)
                else:
                    buf = param_state['momentum_buffer']
                    buf_w = param_state['wmomentum_buffer']
                    gvv += (buf-(d_p+weight_decay*p.data)).pow(2).sum()
                    gn += buf.pow(2).sum()
                    pn += torch.numel(buf)
                # d_p_w = d_p_w.mul_(momentum).add_(buf)  # TODO: why had this?
                # if self.opt.wmomentum:
                #     d_p = d_p_w
                if (self.epoch < self.opt.g_update_start
                        or self.epoch % self.opt.g_update_interval == 0):
                    buf.mul_(momentum).add_(d_p)
                buf_w.mul_(wmomentum).add_(d_p_w)
                d_p_w = buf_w
                d_p = buf
                # d_p_w = d_p_w.add(momentum, buf_w)
                # buf.mul_(momentum).add_(d_p)
                # d_p = buf
                if self.opt.sampler:
                    p.data.add_(-group['lr'], d_p)
                else:
                    p.data.add_(-group['lr'], d_p_w)
            self.profiler.toc('step')

        self.logger.update('normg_sub_normg', normg_ratio[:len(idx)], len(idx))
        self.logger.update('dmom_sub_dmom', dmom_ratio[:len(idx)], len(idx))
        self.logger.update('newdmom_div_normg',
                           newdmom_normg_ratio[:len(idx)], len(idx))
        self.logger.update('numz', numz, len(idx))
        self.logger.update('zpercent', numz*100./len(idx), len(idx))
        self.logger.update('bigg', bigg, len(idx))
        self.logger.update('normg', normg_val[:len(idx)], len(idx))
        self.logger.update('dmom', dmom_val[:len(idx)], len(idx))
        self.logger.update('overflow', numo, len(idx))
        self.logger.update('alpha', alpha_val[:len(idx)], len(idx))
        self.logger.update('alpha_sq_sum', float((alpha_val*alpha_val).sum()),
                           len(idx))
        self.logger.update('alpha_sum', float(alpha_val.sum()), len(idx))
        self.logger.update('normg_sum', float(normg_val.sum()), len(idx))
        self.logger.update('normg_sq_sum', float((normg_val*normg_val).sum()),
                           len(idx))
        self.logger.update('alpha__normg_sq_sum',
                           float((alpha_val*normg_val*normg_val).sum()),
                           len(idx))
        self.logger.update('zp_sample', nz_sample*100./len(idx), len(idx))
        self.logger.update('sampler_alpha_th', sampler_alpha_th, len(idx))
        self.logger.update('gw_variance_mean', gvw_mean, len(idx))
        self.logger.update('g_variance_mean', gv_mean, len(idx))
        self.logger.update('gw_diff_g', float(np.sqrt(gw_g)), len(idx))
        self.logger.update('gw_norm', gw_norm, len(idx))
        self.logger.update('g_norm', g_norm, len(idx))
        self.logger.update('grad_var', gvv/(pn + 1e-7), 1)
        self.logger.update('grad_var_n', gvv/(gn + 1e-7), 1)
        self.logger.update('gbar_norm', gn, 1)

    def log_perc(self, prefix=''):
        self.logger.update(prefix+'dmom_h', self.data_momentum, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'normg_h', self.grad_norm, 1, hist=True,
                           log_scale=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            dmvn = np.divide(self.data_momentum, self.grad_norm)
        self.logger.update(prefix+'dmom_div_normg_h', dmvn, 1, hist=True,
                           log_scale=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            avn = np.divide(self.alpha, self.grad_norm)
        self.logger.update(prefix+'alpha_div_normg_h', avn, 1, hist=True,
                           log_scale=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            nva = np.divide(self.grad_norm, self.alpha)
        self.logger.update(prefix+'normg_div_alpha_h', avn, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'avn_p_nva_h', avn+nva, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'fnorm_h', self.fnorm, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'alpha_h', self.alpha, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'alpha_normed_h',
                           self.alpha_normed, 1, hist=True, log_scale=True)
        self.logger.update(prefix+'loss_h', self.loss, 1, hist=True,
                           log_scale=True)
        if self.alpha_normed_pre is not None:
            # iou = np.zeros(10);
            # for i in range(1, 10):
            #     sc = set(np.where(self.alpha_normed > np.exp(-10))[0])
            #     sp = set(np.where(self.alpha_normed_pre > np.exp(-10))[0])
            #     iou[i] = len(sc.intersection(sp))/len(sc+sp)
            sc = len(self.alpha_normed)-1-np.argsort(self.alpha_normed)
            sp = len(self.alpha_normed)-1-np.argsort(self.alpha_normed_pre)
            sa = np.maximum(sc, sp)
            saf = sa[np.where(sa < sa.size/10)[0]]
            # TODO: non-log perc
            self.logger.update(prefix+'big_alpha_vs_pre_h', saf, 1, hist=True)
        self.alpha_normed_pre = self.alpha_normed
