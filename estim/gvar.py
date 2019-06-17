import numpy as np
import torch
import torch.nn
import torch.multiprocessing

from estim.sgd import SGDEstimator
from estim.gluster import GlusterOnlineEstimator, GlusterBatchEstimator
from estim.svrg import SVRGEstimator
from estim.ntk import NeuralTangentKernelEstimator, NeuralTangentKernelFull
from estim.kfac import KFACEstimator
from estim.kfac_orig import KFACEstimatorOrig
from estim.bf_fisher import BruteForceFisher, BruteForceFisherFull
from estim.lanczos import LanczosEstimator

import optim
import optim.adamw
import optim.kfac
import optim.kfac_orig
import optim.ekfac
import utils


def init_estimator(g_estim, opm, data_loader, opt, tb_logger):
    # gluster or SVRG
    if g_estim == 'gluster':
        if opt.g_online:
            gest = GlusterOnlineEstimator(
                    data_loader, opt, tb_logger)
        else:
            gest = GlusterBatchEstimator(
                    data_loader, opt, tb_logger)
    elif g_estim == 'svrg':
        gest = SVRGEstimator(data_loader, opt, tb_logger)
    elif g_estim == 'sgd':
        gest = SGDEstimator(data_loader, opt, tb_logger)
    elif g_estim == 'ntk':
        gest = NeuralTangentKernelEstimator(data_loader, opt, tb_logger)
    elif g_estim == 'ntkf':
        gest = NeuralTangentKernelFull(data_loader, opt, tb_logger)
    elif g_estim == 'kfaco':
        gest = KFACEstimatorOrig(opm, data_loader, opt, tb_logger)
    elif g_estim == 'kfac' or g_estim == 'kface':
        empirical = g_estim == 'kface'
        gest = KFACEstimator(opm, empirical, data_loader, opt, tb_logger)
    elif g_estim == 'bffisher':
        gest = BruteForceFisher(data_loader, opt, tb_logger)
    elif g_estim == 'bffisherf':
        gest = BruteForceFisherFull(data_loader, opt, tb_logger)
    elif g_estim == 'lanczos':
        gest = LanczosEstimator(data_loader, opt, tb_logger)
    return gest


def init_optim(optim_name, model, opt):
    if optim_name == 'sgd' or optim_name == 'ntk':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt.lr, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=opt.nesterov)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     betas=opt.adam_betas,
                                     eps=opt.adam_eps,
                                     weight_decay=opt.weight_decay)
    elif optim_name == 'adamw':
        optimizer = optim.adamw.AdamW(
            model.parameters(),
            lr=opt.lr,
            betas=opt.adam_betas,
            eps=opt.adam_eps,
            weight_decay=opt.weight_decay,
            l2_reg=False)
    elif optim_name == 'kfac':
        optimizer = optim.kfac.KFACOptimizer(
            model,
            lr=opt.lr,
            momentum=opt.momentum,
            stat_decay=opt.kf_stat_decay,
            damping=opt.kf_damping,
            kl_clip=opt.kf_kl_clip,
            weight_decay=opt.weight_decay,
            TCov=opt.kf_TCov,
            TInv=opt.kf_TInv)
    elif optim_name == 'kfaco':
        optimizer = optim.kfac_orig.KFACOptimizerOrig(
            model,
            lr=opt.lr,
            momentum=opt.momentum,
            stat_decay=opt.kf_stat_decay,
            damping=opt.kf_damping,
            kl_clip=opt.kf_kl_clip,
            weight_decay=opt.weight_decay,
            TCov=opt.kf_TCov,
            TInv=opt.kf_TInv)
        model.criterion.optim = optimizer
    elif optim_name == 'ekfac':
        optimizer = optim.ekfac.EKFACOptimizer(
            model,
            lr=opt.lr,
            momentum=opt.momentum,
            stat_decay=opt.kf_stat_decay,
            damping=opt.kf_damping,
            kl_clip=opt.kf_kl_clip,
            weight_decay=opt.weight_decay,
            TCov=opt.kf_TCov,
            TScal=opt.kf_TScal,
            TInv=opt.kf_TInv,
            use_gestim=(not opt.kf_gestim))
        if opt.kf_gestim:
            model.criterion.optim = optimizer
    return optimizer


class GEstimatorCollection(object):
    def __init__(self, model, data_loader, opt, tb_logger):
        self.gest_used = False
        self.optim = []
        self.gest = []
        self.opt = opt
        ges = opt.g_estim.split(',')
        for optim_name in opt.optim.split(','):
            opm = init_optim(optim_name, model, opt)
            opm.secondary_optim = len(self.optim) != 0
            self.optim += [(optim_name, opm)]
        for eid, g_estim in enumerate(ges):
            opm = (self.optim[0][1]
                   if len(self.optim) == 1
                   else self.optim[eid][1])
            self.gest += [(g_estim, init_estimator(
                g_estim, opm, data_loader, opt, tb_logger))]
        self.niters = 0

    def update_niters(self, niters):
        self.niters = niters
        for name, estim in self.gest:
            estim.update_niters(self.niters)
        for name, opm in self.optim:
            opm.steps = niters

    def snap_batch(self, model):
        for name, gest in self.gest:
            if 'kfac' in name:
                gest.snap_batch(model)

    def snap_online(self, model):
        for name, gest in self.gest:
            if 'kfac' in name:
                gest.snap_online(model)

    def snap_model(self, model):
        for name, gest in self.gest:
            gest.snap_model(model)

    def log_var(self, model, gviter, tb_logger):
        niters = self.niters
        Ege_s = []
        bias_str = ''
        var_str = ''
        snr_str = ''
        nvar_str = ''
        evals = []
        estim_str = '%s' % self.gest[self.gest_used][0]
        keys = []
        vals = []
        data = None
        for i, (name, gest) in enumerate(self.gest):
            Ege, var_e, snr_e, nv_e = gest.get_Ege_var(model, gviter)
            if self.opt.log_eigs:
                ev, data = gest.get_precond_eigs(model, data=data)
                evals += [ev.sort(descending=True)[0].cpu().numpy()]
            if i == 0:
                tb_logger.log_value('sgd_var', float(var_e), step=niters)
                tb_logger.log_value('sgd_snr', float(snr_e), step=niters)
                tb_logger.log_value('sgd_nvar', float(nv_e), step=niters)
            if i == 1:
                tb_logger.log_value('est_var', float(var_e), step=niters)
                tb_logger.log_value('est_snr', float(snr_e), step=niters)
                tb_logger.log_value('est_nvar', float(nv_e), step=niters)
            tb_logger.log_value('est_var%d' % i, float(var_e), step=niters)
            tb_logger.log_value('est_snr%d' % i, float(snr_e), step=niters)
            tb_logger.log_value('est_nvar%d' % i, float(nv_e), step=niters)
            Ege_s += [Ege]
            var_str += '%s: %.8f ' % (name, var_e)
            snr_str += '%s: %.8f ' % (name, snr_e)
            nvar_str += '%s: %.8f ' % (name, nv_e)
            if i > 0:
                bias = torch.mean(torch.cat(
                    [(ee-gg).abs().flatten()
                     for ee, gg in zip(Ege_s[0], Ege_s[i])]))
                if i == 1:
                    tb_logger.log_value('grad_bias', float(bias), step=niters)
                tb_logger.log_value('grad_bias%d' % i,
                                    float(bias), step=niters)
                bias_str += '%s: %.8f\t' % (name, bias)
        if self.opt.log_eigs:
            names = [n for n, g in self.gest]
            tb_logger.log_list_of_vectors('eigs', evals, step=niters,
                                          tags=names)
        keys = ['Estim used', 'Bias', 'Var', 'N-Var']
        vals = [estim_str, bias_str, var_str, nvar_str]  # snr_str
        if self.opt.log_eigs and len(self.gest) > 1:
            n = min(len(evals[-1]), len(evals[-2]))
            eval_diff = np.mean(np.abs(evals[-1][-n:]-evals[-2][-n:]))
            tb_logger.log_value('eigs_diff', float(eval_diff), step=niters)
            keys += ['E-vals(n=%d,m=%d)' % (n, len(evals[-1] > 1e-5))]
            vals += ['%.8f' % eval_diff]
        s = ''
        for k, v in zip(keys, vals):
            s += '%s: %s\t\t' % (k, v)
        return s

    def grad(self, use_sgd, *args, **kwargs):
        self.gest_used = not use_sgd
        return self.get_estim().grad(*args, **kwargs)

    def state_dict(self):
        return (self.get_estim().state_dict(),
                self.get_optim().state_dict())

    def load_state_dict(self, state):
        self.get_estim().load_state_dict(state[0])
        self.get_optim().load_state_dict(state[1])

    def get_estim(self):
        estim_id = self.gest_used if len(self.gest) > 1 else 0
        return self.gest[estim_id][1]

    def get_optim(self):
        optim_id = self.gest_used if len(self.optim) > 1 else 0
        return self.optim[optim_id][1]


class MinVarianceGradient(object):
    def __init__(self, model, data_loader, opt, tb_logger):
        self.model = model
        self.gest = GEstimatorCollection(model, data_loader, opt, tb_logger)
        self.opt = opt
        self.tb_logger = tb_logger
        self.init_snapshot = False
        self.gest_used = False
        self.gest_counter = 0
        self.Esgd = 0
        self.last_log_iter = 0
        self.opt = opt
        self.niters = 0

    def update_niters(self, niters):
        self.niters = niters
        self.gest.update_niters(niters)

    def is_log_iter(self):
        niters = self.niters
        opt = self.opt
        if (niters-self.last_log_iter >= opt.gvar_log_iter
                and niters >= opt.gvar_start):
            self.last_log_iter = niters
            return True
        return False

    def snap_batch(self, model):
        # model.eval()  # done inside SVRG
        model.train()
        # Cosine sim
        # gviter = self.opt.gvar_estim_iter
        # self.Esgd = self.sgd.get_Ege_var(model, gviter)[0]
        # if ((self.niters - self.opt.gvar_start) % self.opt.g_bsnap_iter == 0
        #         and self.niters >= self.opt.gvar_start):
        self.gest.snap_batch(model)
        self.init_snapshot = True
        self.gest_counter = 0

    def snap_online(self, model):
        # model.eval()  # TODO: keep train
        model.train()
        # if ((self.niters - self.opt.gvar_start) % self.opt.g_osnap_iter == 0
        #     and self.niters >= self.opt.gvar_start):
        self.gest.snap_online(model)

    def snap_model(self, model):
        # if ((self.niters % self.opt.g_msnap_iter == 0 and self.opt.g_avg > 1)
        #         or (self.niters == 0 and self.opt.g_avg == 1)):
        self.gest.snap_model(model)

    def log_var(self, model):
        tb_logger = self.tb_logger
        # model.eval()
        # if not self.init_snapshot:
        #     return ''
        gviter = self.opt.gvar_estim_iter
        return self.gest.log_var(model, gviter, tb_logger)

    def grad(self):
        model = self.model
        model.train()
        use_sgd = self.use_sgd(self.niters)
        self.gest_used = not use_sgd
        if self.gest_used:
            self.gest_counter += 1
        return self.gest.grad(use_sgd, model, in_place=True)

    def use_sgd(self, niters):
        use_sgd = not self.opt.g_optim or niters < self.opt.g_optim_start
        use_sgd = use_sgd or (self.opt.g_optim_max > 0 and
                              self.gest_counter >= self.opt.g_optim_max)
        if self.gest_used != (not use_sgd):
            self.gest.get_optim().niters = niters
            utils.adjust_lr(self.gest.get_optim(), self.opt)
        return use_sgd

    @property
    def param_groups(self):
        return self.gest.get_optim().param_groups

    def state_dict(self):
        return self.gest.state_dict()

    def load_state_dict(self, state):
        self.gest.load_state_dict(state)
        self.init_snapshot = True

    def zero_grad(self):
        return self.gest.get_optim().zero_grad()

    def step(self):
        return self.gest.get_optim().step()
