import torch
import torch.nn
import torch.multiprocessing
import torch.nn.functional as F

from .sgd import SGDEstimator


class KFACEstimator(SGDEstimator):
    def __init__(self, opm, empirical=False, *args, **kwargs):
        super(KFACEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.optim = opm
        self.acc_stats_init = True
        self.optim.acc_stats = False
        self.empirical = empirical
        self.snap_one = self.opt.kf_snap_one

    def update_niters(self, niters):
        self.niters = niters
        # self.optim.steps = niters

    def grad(self, model, in_place=False, data=None):
        if data is None:
            data = next(self.data_iter)

        model.zero_grad()
        if self.snap_one:
            loss = self.snap_online0(model, data)
            self.optim.update_inv()
            loss.backward(retain_graph=True)
        else:
            loss = model.criterion(model, data)
            loss.backward(retain_graph=True)

        if in_place:
            # loss.backward()
            self.optim.apply_precond()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        self.optim.apply_precond_outplace(g)
        return g

    def snap_TCov(self, loss, output, target):
        # compute true fisher
        self.optim.acc_stats = True
        with torch.no_grad():
            if self.empirical:
                # sampled_y = output.max(1)[1]
                sampled_y = target
            else:
                probs = torch.nn.functional.softmax(output, dim=1)
                sampled_y = torch.multinomial(probs, 1).squeeze()
        loss_sample = F.nll_loss(output, sampled_y, reduction='mean')
        # loss_sample = F.nll_loss(output, sampled_y, reduction='none').sum()
        # self.batch_size = target.shape[0]
        loss_sample.backward(retain_graph=True)
        self.optim.acc_stats = False

    def snap_online0(self, model, data):
        self.optim.active = True
        model.zero_grad()
        loss, output = model.criterion(model, data, return_output=True)
        self.snap_TCov(loss, output, data[1].cuda())
        model.zero_grad()  # clear the gradient for computing true-fisher.
        self.optim.active = False
        return loss

    def snap_online(self, model):
        if self.snap_one:
            return
        if self.niters % self.opt.kf_TCov == 0 or self.acc_stats_init:
            self.acc_stats_init = False
            data = next(self.data_iter)
            self.snap_online0(model, data)

    def snap_batch(self, model):
        if self.snap_one:
            return
        if self.niters % self.optim.TInv == 0:
            self.optim.update_inv()

    def get_precond_eigs(self, *args, **kwargs):
        self.optim.save = True
        ret = super(KFACEstimator, self).get_precond_eigs(*args, **kwargs)
        self.optim.save = False
        return ret

    def get_precond_eigs_nodata(self):
        # return self.optim.get_precond_eigs()/self.batch_size
        return self.optim.get_precond_eigs()
