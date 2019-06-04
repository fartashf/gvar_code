import logging

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
        minvar_loader = get_minvar_loader(train_loader, opt)
        self.gvar = MinVarianceGradient(model, minvar_loader, opt, tb_logger)

    @property
    def param_groups(self):
        return self.gvar.param_groups

    def step(self, profiler):
        gvar = self.gvar
        opt = self.opt
        model = self.model

        self.gvar.zero_grad()

        pg_used = gvar.gest_used
        loss = gvar.grad(self.niters)
        if gvar.gest_used != pg_used:
            logging.info('Optimizer changed.')
        self.gvar.step()
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
