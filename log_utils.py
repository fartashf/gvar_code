from collections import OrderedDict
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return '%d' % self.val
        return '%.4f (%.4f)' % (self.val, self.avg)

    def tb_log(self, tb_logger, name, step=None):
        tb_logger.log_value(name, self.val, step=step)


class StatisticMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mu = AverageMeter()
        self.std = AverageMeter()
        self.min = AverageMeter()
        self.max = AverageMeter()
        self.med = AverageMeter()

    def update(self, val, n=0):
        val = np.ma.masked_invalid(val)
        val = val.compressed()
        n = min(n, len(val))
        if n == 0:
            return
        self.mu.update(np.mean(val), n=n)
        self.std.update(np.std(val), n=n)
        self.min.update(np.min(val), n=n)
        self.max.update(np.max(val), n=n)
        self.med.update(np.median(val), n=n)

    def __str__(self):
        # return 'mu:{}|med:{}|std:{}|min:{}|max:{}'.format(
        #     self.mu, self.med, self.std, self.min, self.max)
        return 'mu:{}|med:{}'.format(self.mu, self.med)

    def tb_log(self, tb_logger, name, step=None):
        self.mu.tb_log(tb_logger, name+'_mu', step=step)
        self.med.tb_log(tb_logger, name+'_med', step=step)
        self.std.tb_log(tb_logger, name+'_std', step=step)
        self.min.tb_log(tb_logger, name+'_min', step=step)
        self.max.tb_log(tb_logger, name+'_max', step=step)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        if k not in self.meters:
            if type(v).__module__ == np.__name__:
                self.meters[k] = StatisticMeter()
            else:
                self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k+': '+str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.iteritems():
            v.tb_log(tb_logger, prefix+k, step=step)
