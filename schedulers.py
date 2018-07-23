import numpy as np


def get_scheduler(train_size, opt):
    sched_dict = {'linrank': LinRankScheduler,
                  'exp_snooze_th': ExpSnoozerThresh,
                  'expsnz_tau': ExpSnoozerThresh,
                  'exp_snooze_med': ExpSnoozerMedian,
                  'exp_snooze_mean': ExpSnoozerMean,
                  'exp_snooze_perc': ExpSnoozerPerc,
                  'exp_snooze_lin': ExpSnoozerLinearRank,
                  'exp_snooze_expdec': ExpSnoozerDecInc,
                  'exp_snooze_stepdec': ExpSnoozerDecInc,
                  'exp_snooze_expinc': ExpSnoozerDecInc,
                  'exp_snooze_stepinc': ExpSnoozerDecInc,
                  'exp_snooze_tauXstd': ExpSnoozerThreshStd,
                  'expsnz_tauXstd': ExpSnoozerThreshStd,
                  'exp_snooze_tauXstdL': ExpSnoozerThreshStdLog,
                  'expsnz_tauXstdL': ExpSnoozerThreshStdLog,
                  'expsnz_cumsum': ExpSnoozerCumSum,
                  'expsnz_tauXmu': ExpSnoozerThreshMean,
                  'expsnz_tauXstdLB': ExpSnoozerThreshStdLogBiased,
                  }
    return sched_dict[opt.sampler_w2c](train_size, opt)


class DMomScheduler(object):
    def __init__(self, train_size, opt):
        self.opt = opt
        self.alpha_mom = np.zeros(train_size)
        self.alpha_mom_bc = np.zeros(train_size)
        self.alpha = np.ones((train_size,))
        self.alpha_normed = np.ones((train_size,))
        self.alpha_normed_pre = None
        self.loss = np.ones((train_size,))
        self.epoch = 0
        self.target = np.zeros((train_size,))

        num_samples = train_size
        self.num_samples = num_samples
        self.indices = range(num_samples)
        self.weights = np.ones(num_samples)
        self.visits = np.zeros(num_samples)
        self.niters = 0
        self.maxiter = 0
        self.epoch_iters = int(np.ceil(1.*self.num_samples/opt.batch_size))

        self.logger = None
        self.abias = 0

    def next_epoch(self):
        I = range(self.num_samples)
        if ((self.opt.scheduler or self.opt.sampler)
                and self.epoch >= self.opt.sampler_start_epoch):
            self.indices = self.schedule()
            print(len(self.indices))
            # print(np.histogram(self.visits, bins='doane')[0])
            # print(map(int, np.histogram(self.visits, bins='doane')[1]))
            I = self.indices
        self.epoch += 1
        return I

    def schedule(self):
        raise NotImplemented("schedule not implemented.")

    def update_alpha(self, idx, alpha):
        self.visits[idx] += 1

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

        alpha_norm = self.normalize_alpha(alpha_bc)
        self.alpha_normed[idx] = alpha_norm

    def compute_dmom(self, alpha, idx):
        dmom = self.opt.dmom
        # alpha_mom
        alpha_mom = self.alpha_mom[idx]*dmom + alpha*(1-dmom)

        # bias correction (after storing)
        alpha_bc = alpha_mom/(1-dmom**self.visits[idx])  # self.epochs+1

        return alpha_mom, alpha_bc

    def normalize_alpha(self, alpha_val):
        # normalize alphas to sum to 1
        if self.opt.alpha_norm == 'sum':
            alpha_val /= alpha_val.sum()
        elif self.opt.alpha_norm == 'bs':
            alpha_val /= len(alpha_val)
        return alpha_val

    def log_perc(self, prefix=''):
        self.logger.update(prefix+'dmom_h', self.alpha_mom, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'alpha_h', self.alpha, 1, hist=True,
                           log_scale=True)
        self.logger.update(prefix+'alpha_normed_h',
                           self.alpha_normed, 1, hist=True, log_scale=True)
        self.logger.update(prefix+'sloss_h', self.loss, 1, hist=True,
                           log_scale=True)
        if self.alpha_normed_pre is not None:
            sc = len(self.alpha_normed)-1-np.argsort(self.alpha_normed)
            sp = len(self.alpha_normed)-1-np.argsort(self.alpha_normed_pre)
            sa = np.maximum(sc, sp)
            saf = sa[np.where(sa < sa.size/10)[0]]
            self.logger.update(prefix+'big_alpha_vs_pre_h', saf, 1, hist=True)
        self.alpha_normed_pre = self.alpha_normed
        self.logger.update(prefix+'alpha_normed_biased_h',
                           self.alpha_normed-self.abias,
                           1, hist=True, log_scale=True)

        count = self.count
        count_nz = float((count == 0).sum())
        self.logger.update(prefix+'touch_p', count_nz*100./len(count), 1)
        self.logger.update(prefix+'touch', count_nz, 1)
        self.logger.update(prefix+'count_h', count, 1, hist=True)
        self.logger.update(prefix+'weights_h', self.weights, 1, hist=True)
        self.logger.update(prefix+'visits_h', self.visits, 1, hist=True)
        closs_h = class_loss(self.loss, self.target)
        self.logger.update(prefix+'slossC_h', closs_h, 1, hist=True,
                           log_scale=True, bins=min(len(closs_h), 100))


def class_loss(loss, target):
    nclasses = int(target.max()+1)
    closs_h = np.zeros(nclasses)
    for i in range(nclasses):
        closs_h[i] = loss[target == i].sum()
    return closs_h


class LinRankScheduler(DMomScheduler):
    def __init__(self, train_size, opt):
        super(LinRankScheduler, self).__init__(train_size, opt)
        num_samples = train_size
        self.count = np.zeros(num_samples)
        self.last_count = np.zeros(num_samples)
        params = map(int, self.opt.sampler_params.split(','))
        self.perc = params[0]
        self.rank = np.ones(num_samples)*num_samples

    def schedule(self):
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        niters = self.niters
        params = map(int, self.opt.sampler_params.split(','))
        perc_a, perc_b, delay_a, delay_b = params
        epoch_iters = self.opt.maxiter
        perc = (1. * niters / epoch_iters) * (perc_b - perc_a) + perc_a
        count = self.count
        visited = (count == 0)
        revisits = (count == 1)

        ids = np.arange(len(alpha))
        np.random.shuffle(ids)
        rank = np.zeros(len(alpha))
        alpha_x = np.array(alpha)
        alpha_x[revisits] = alpha.max() + 1
        srti = np.argsort(alpha_x[ids])
        rank[ids[srti]] = np.arange(len(alpha))

        drop_rank = int(self.num_samples * perc / 100)
        ids_d = ids[srti[:drop_rank]]
        ids_r = ids[srti[drop_rank:]]
        tobevisited = np.zeros(self.num_samples, dtype=np.bool)
        tobevisited[ids_r] = True
        count[count > 1] -= 1  # not the revisits that won't be visited
        # count[:] = 3
        count[tobevisited] = 0

        needdelay = np.logical_and(visited, np.logical_not(tobevisited))
        delay = (drop_rank-rank[needdelay])/drop_rank * niters/epoch_iters
        delay = (1. * delay) * (delay_b - delay_a) + delay_a
        count[needdelay] = np.int32(np.ceil(delay))
        print('Delay perc: %d delay: %d' % (perc, count[ids_d].mean()))
        tau = alpha_x[ids_d].max()
        self.logger.update('tau', float(tau), 1)

        # shared updates
        self.weights[self.indices] = 0
        self.weights = np.minimum(self.opt.sampler_maxw, self.weights + 1)
        self.count = np.around(count - np.min(count))  # min should be 0
        while ((self.count == 0).sum() <
               self.opt.min_batches * self.opt.batch_size):
            self.count = np.maximum(0, self.count - 1)
        return np.where(self.count == 0)[0]


class ExpSnoozer(DMomScheduler):
    def __init__(self, train_size, opt):
        super(ExpSnoozer, self).__init__(train_size, opt)
        self.delta = np.zeros(train_size)
        self.wait = np.zeros(train_size)
        self.active = np.zeros(train_size, dtype=np.bool)
        self.count = np.zeros(train_size)
        self.num_bumped = 0

    def schedule_tau(self, tau):
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        awake_cond = alpha >= tau
        tau = float(tau)
        self.logger.update('tau', tau, 1)

        awake = np.logical_and(self.active, awake_cond)
        snoozed = np.logical_and(self.active, np.logical_not(awake_cond))
        bumped = np.logical_and(awake, self.delta > 0)
        self.num_bumped = nb = bumped.sum()
        self.delta[awake] = 0
        self.delta[snoozed] += 1
        self.logger.update('bumped', float(nb), 1)
        self.logger.update('bumped_p', float(nb*100./self.num_samples), 1)

        # shared updates
        self.weights[:] = 1
        self.active[:] = False
        while self.active.sum() < self.opt.min_batches * self.opt.batch_size:
            self.wait[np.logical_not(self.active)] += 1
            self.active[np.power(2, self.delta) == self.wait] = True
        self.count[:] = np.power(2, self.delta) - self.wait
        self.wait[self.active] = 0
        na = (1-self.active).sum()
        self.logger.update('snooze', float(na), 1)
        self.logger.update('snooze_p', float(na*100./self.num_samples), 1)
        return np.where(self.active)[0]


class ExpSnoozerThresh(ExpSnoozer):
    def schedule(self):
        tau = float(self.opt.sampler_params)
        return self.schedule_tau(tau)


class ExpSnoozerMedian(ExpSnoozer):
    def schedule(self):
        frac = float(self.opt.sampler_params)
        srti = np.argsort(self.alpha_normed)
        tau = frac*self.alpha_normed[srti[len(srti)//2]]
        return self.schedule_tau(tau)


class ExpSnoozerMean(ExpSnoozer):
    def schedule(self):
        frac = float(self.opt.sampler_params)
        tau = frac*self.alpha_normed.mean()
        return self.schedule_tau(tau)


class ExpSnoozerPerc(ExpSnoozer):
    def schedule(self):
        perc, frac = map(float, self.opt.sampler_params.split(','))
        tau = frac*np.percentile(self.alpha_normed, perc)
        return self.schedule_tau(tau)


class ExpSnoozerLinearRank(ExpSnoozer):
    def __init__(self, train_size, opt):
        super(ExpSnoozerLinearRank, self).__init__(train_size, opt)
        self.perc, self.stepi = map(float, self.opt.sampler_params.split(','))
        self.step = 1

    def schedule(self):
        tau = np.percentile(self.alpha_normed, self.perc)
        I = self.schedule_tau(tau)
        nb_post = self.num_bumped
        if nb_post > .02*self.num_samples:
            self.perc = max(0, self.perc-self.step)
            self.step = 1
        else:
            self.perc = min(100, self.perc+self.step)
            self.step += self.stepi
        self.logger.update('snooze_step', float(self.step), 1)
        return I


class ExpSnoozerDecInc(ExpSnoozer):
    def __init__(self, train_size, opt):
        super(ExpSnoozerDecInc, self).__init__(train_size, opt)
        params = map(float, self.opt.sampler_params.split(','))
        self.tau, self.decay_epoch = params

    def schedule(self):
        epoch = float(self.niters//self.epoch_iters)
        if (self.opt.sampler_w2c.endswith('stepdec') or
                self.opt.sampler_w2c.endswith('stepinc')):
            epoch = int(epoch)
        last_epoch = 2. ** (epoch / int(self.decay_epoch)) - 1
        base = 0.1 if self.opt.sampler_w2c.endswith('dec') else 10.
        tau = self.tau * (base ** last_epoch)
        return self.schedule_tau(tau)


class ExpSnoozerThreshStd(ExpSnoozer):
    def schedule(self):
        tau = float(self.opt.sampler_params)
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        mu = np.mean(alpha[alpha > np.exp(-20)])
        std = np.std(alpha[alpha > np.exp(-20)])
        # awake_cond = np.logical_and(alpha >= mu-tau*std,
        #                             alpha <= mu+tau*std)
        self.logger.update('tauloss_mu', float(mu), 1)
        self.logger.update('tauloss_std', float(std), 1)
        tau = mu-tau*std
        return self.schedule_tau(tau)


class ExpSnoozerThreshStdLog(ExpSnoozer):
    def schedule(self):
        tau = float(self.opt.sampler_params)
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        mu = np.mean(np.log(alpha[alpha > np.exp(-20)]))
        std = np.std(np.log(alpha[alpha > np.exp(-20)]))
        self.logger.update('tauloss_mu', float(mu), 1)
        self.logger.update('tauloss_std', float(std), 1)
        tau = mu-tau*std
        tau = np.exp(tau)
        return self.schedule_tau(tau)


class ExpSnoozerCumSum(ExpSnoozer):
    def schedule(self):
        tau = float(self.opt.sampler_params)
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        alpha_cs = np.cumsum(np.sort(alpha))
        idx = np.argmin(np.abs(alpha_cs - alpha_cs[-1]*tau))
        tau = alpha[idx]
        return self.schedule_tau(tau)


class ExpSnoozerThreshMean(ExpSnoozer):
    def schedule(self):
        tau = float(self.opt.sampler_params)
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        mu = np.mean(alpha[alpha > np.exp(-20)])
        tau = tau*mu
        return self.schedule_tau(tau)


class ExpSnoozerThreshStdLogBiased(ExpSnoozer):
    def schedule(self):
        tau = float(self.opt.sampler_params)
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        self.abias = np.percentile(alpha, 10)/10
        print(self.abias)
        alpha -= self.abias
        mu = np.mean(np.log(alpha[alpha > np.exp(-20)]))
        std = np.std(np.log(alpha[alpha > np.exp(-20)]))
        self.logger.update('tauloss_mu', float(mu), 1)
        self.logger.update('tauloss_std', float(std), 1)
        tau = mu-tau*std
        tau = self.abias+np.exp(tau)
        return self.schedule_tau(tau)
