import numpy as np


class DMomScheduler(object):
    def __init__(self, train_size, opt):
        self.opt = opt
        self.alpha_mom = np.zeros(train_size)
        self.alpha_mom_bc = np.zeros(train_size)
        self.alpha = np.ones((train_size,))
        self.alpha_normed = np.ones((train_size,))
        self.loss = np.ones((train_size,))
        self.epoch = 0

        num_samples = train_size
        self.num_samples = num_samples
        self.count = np.zeros(num_samples)
        self.indices = range(num_samples)
        self.weights = np.ones(num_samples)
        self.visits = np.zeros(num_samples)
        self.last_count = np.zeros(num_samples)
        params = map(int, self.opt.sampler_params.split(','))
        self.perc = params[0]
        self.rank = np.ones(num_samples)*num_samples
        self.niters = 0

    def next_epoch(self):
        I = range(self.num_samples)
        if self.epoch >= self.opt.sampler_start_epoch:
            self.schedule()
            self.indices = np.where(self.count == 0)[0]
            print(len(self.indices))
            # print(np.histogram(self.visits, bins='doane')[0])
            # print(map(int, np.histogram(self.visits, bins='doane')[1]))
            I = self.indices
        self.epoch += 1
        return I

    def schedule(self):
        opt = self.opt
        alpha = np.array(self.alpha_normed, dtype=np.float32)
        if self.opt.sampler_w2c == 'linrank':
            niters = self.niters
            params = map(int, self.opt.sampler_params.split(','))
            perc_a, perc_b, delay_a, delay_b = params
            epoch_iters = opt.maxiter
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
        self.weights[self.indices] = 0
        self.weights = np.minimum(self.opt.sampler_maxw, self.weights + 1)
        self.count = np.around(count - np.min(count))  # min should be 0
        while (self.count == 0).sum() < self.opt.batch_size:
            self.count = np.maximum(0, self.count - 1)

    def update_alpha(self, idx, alpha):
        self.visits[idx] += 1
        self.alpha[idx] = alpha

        alpha_mom, alpha_bc = self.compute_dmom(alpha, idx)
        self.alpha_mom[idx] = alpha_mom
        self.alpha_mom_bc[idx] = alpha_bc

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
        elif self.opt.alpha_norm == 'none':
            alpha_val /= len(alpha_val)
        return alpha_val

