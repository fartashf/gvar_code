import logging
import torch

import optim
import optim.adamw
import optim.kfac
import optim.ekfac

import utils
from data import get_minvar_loader
from log_utils import LogCollector
from estim.gvar import MinVarianceGradient


class OptimizerFactory(object):

    def __init__(self, model, train_loader, tb_logger, opt):
        self.model = model
        self.opt = opt
        self.niters = 0
        self.optimizer = None
        self.logger = LogCollector(opt)
        self.param_groups = None
        self.gest_used = False
        minvar_loader = get_minvar_loader(train_loader, opt)
        self.gvar = MinVarianceGradient(model, minvar_loader, opt, tb_logger)
        self.reset()

    def reset(self):
        model = self.model
        opt = self.opt
        if opt.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=opt.lr, momentum=opt.momentum,
                                        weight_decay=opt.weight_decay,
                                        nesterov=opt.nesterov)
        elif opt.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=opt.lr,
                                         betas=opt.adam_betas,
                                         eps=opt.adam_eps,
                                         weight_decay=opt.weight_decay)
        elif opt.optim == 'adamw':
            optimizer = optim.adamw.AdamW(
                model.parameters(),
                lr=opt.lr,
                betas=opt.adam_betas,
                eps=opt.adam_eps,
                weight_decay=opt.weight_decay,
                l2_reg=False)
        elif opt.optim == 'kfac':
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
            model.criterion.optim = optimizer
        elif opt.optim == 'ekfac':
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
                TInv=opt.kf_TInv)
            model.criterion.optim = optimizer
        self.optimizer = optimizer
        self.gvar.gest.optimizer = optimizer  # TODO: refactor
        if self.param_groups is not None:
            self.optimizer.param_groups = self.param_groups
        else:
            self.param_groups = self.optimizer.param_groups

    def step(self, profiler):
        gvar = self.gvar
        opt = self.opt
        model = self.model

        self.optimizer.zero_grad()

        pg_used = gvar.gest_used
        loss = gvar.grad(self.niters)
        if gvar.gest_used != pg_used:
            logging.info('Optimizer reset.')
            self.gest_used = gvar.gest_used
            utils.adjust_lr(self, opt)
            if not (opt.optim == 'kfac' or opt.optim == 'ekfac'):
                self.reset()
        self.optimizer.step()
        profiler.toc('optim')

        # snapshot
        if ((self.niters % opt.g_msnap_iter == 0 and opt.g_avg > 1)
                or (self.niters == 0 and opt.g_avg == 1)):
            # Update model snaps
            gvar.snap_model(model)
            profiler.toc('snap_model')
        if ((self.niters - opt.gvar_start) % opt.g_osnap_iter == 0
                and self.niters >= opt.gvar_start):
            # Frequent snaps
            gvar.snap_online(model, self.niters)
            profiler.toc('snap_online')
        if ((self.niters - opt.gvar_start) % opt.g_bsnap_iter == 0
                and self.niters >= opt.gvar_start):
            # Rare snaps
            logging.info('Batch Snapshot')
            gvar.snap_batch(model, self.niters)
            profiler.toc('snap_batch')
            profiler.end()
            logging.info('%s' % str(profiler))
        profiler.end()
        return loss
