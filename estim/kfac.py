import torch
import torch.nn
import torch.multiprocessing
import torch.nn.functional as F

from .sgd import SGDEstimator


class KFACEstimator(SGDEstimator):
    def __init__(self, opm, *args, **kwargs):
        super(KFACEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.optim = opm
        self.acc_stats_init = True
        self.optim.acc_stats = False

    def update_niters(self, niters):
        self.niters = niters
        self.optim.steps = niters

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        loss = model.criterion(model, data)

        if in_place:
            loss.backward()
            self.optim.apply_precond()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        self.optim.apply_precond_outplace(g)
        return g

    def snap_TCov(self, loss, output):
        # compute true fisher
        self.optim.acc_stats = True
        with torch.no_grad():
            sampled_y = torch.multinomial(
                torch.nn.functional.softmax(output.cpu().data, dim=1),
                1).squeeze().cuda()
        loss_sample = F.nll_loss(output, sampled_y, reduction='mean')
        loss_sample.backward(retain_graph=True)
        self.optim.acc_stats = False

    def snap_online(self, model):
        if self.optim.steps % self.optim.TCov == 0 or self.acc_stats_init:
            self.acc_stats_init = False
            data = next(self.data_iter)

            model.zero_grad()
            loss, output = model.criterion(model, data, return_output=True)
            self.snap_TCov(loss, output)
            model.zero_grad()  # clear the gradient for computing true-fisher.

    def snap_batch(self, model):
        model.zero_grad()
        data = next(self.data_iter)
        loss = model.criterion(model, data)
        loss.backward()
        self.optim.update_inv()

    def get_precond_eigs_nodata(self):
        return self.optim.get_precond_eigs()


class KFACEstimatorOrigin(SGDEstimator):
    def __init__(self, opm, *args, **kwargs):
        super(KFACEstimatorOrigin, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.optim = opm
        self.acc_stats_init = True
        self.optim.acc_stats = False

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        model.zero_grad()
        loss, output = model.criterion(model, data, return_output=True)
        if self.optim.steps % self.optim.TCov == 0 or self.acc_stats_init:
            self.acc_stats_init = False
            self.snap_TCov(loss, output)
            model.zero_grad()  # clear the gradient for computing true-fisher.

        if in_place:
            loss.backward()
            self.optim.apply_precond()  # only when it will be used for optim
            return loss
        # TODO: grad after precond
        g = torch.autograd.grad(loss, model.parameters())
        return g

    def snap_TCov(self, loss, output):
        # compute true fisher
        self.optim.acc_stats = True
        with torch.no_grad():
            sampled_y = torch.multinomial(
                torch.nn.functional.softmax(output.cpu().data, dim=1),
                1).squeeze().cuda()
        loss_sample = F.nll_loss(output, sampled_y, reduction='mean')
        loss_sample.backward(retain_graph=True)
        self.optim.acc_stats = False
