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
        # opt = self.opt
        model = self.model

        gvar.snap_online(model, self.niters)

        self.gvar.zero_grad()

        pg_used = gvar.gest_used
        loss = gvar.grad(self.niters)
        if gvar.gest_used != pg_used:
            logging.info('Optimizer changed.')
        self.gvar.step()
        profiler.toc('optim')

        # snapshot
        # gvar.snap_model(model)
        # profiler.toc('snap_model')
        # Frequent snaps
        # gvar.snap_online(model, self.niters)
        # profiler.toc('snap_online')
        # Rare snaps
        # logging.info('Batch Snapshot')
        # gvar.snap_batch(model, self.niters)
        # profiler.toc('snap_batch')
        # profiler.end()
        # logging.info('%s' % str(profiler))
        profiler.end()
        return loss
