import torch
import torch.nn
import torch.multiprocessing
import copy

from data import InfiniteLoader


class GradientEstimator(object):
    def __init__(self, data_loader, opt, tb_logger=None, *args, **kwargs):
        self.opt = opt
        self.model = None
        self.data_loader = data_loader
        self.tb_logger = tb_logger

    def init_data_iter(self):
        self.data_iter = iter(InfiniteLoader(self.data_loader))
        self.estim_iter = iter(InfiniteLoader(self.data_loader))

    def snap_batch(self, model, niters):
        pass

    def update_sampler(self):
        pass

    def snap_online(self, model, niters):
        pass

    def grad(self, model_new, in_place=False):
        raise NotImplemented('grad not implemented')

    def grad_estim(self, model):
        # insuring continuity of data seen in training
        # TODO: make sure sub-classes never use any other data_iter, e.g. raw
        dt = self.data_iter
        self.data_iter = self.estim_iter
        ret = self.grad(model)
        self.data_iter = dt
        return ret

    def get_Ege_var(self, model, gviter):
        # estimate grad mean and variance
        Ege = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad_estim(model)
            for e, g in zip(Ege, ge):
                e += g

        for e in Ege:
            e /= gviter
        nw = sum([w.numel() for w in model.parameters()])
        var_e = 0
        Es = [torch.zeros_like(g) for g in model.parameters()]
        En = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad_estim(model)
            # TODO: is variance important?
            v = sum([(gg-ee).pow(2).sum() for ee, gg in zip(Ege, ge)])
            for s, e, g, n in zip(Es, Ege, ge, En):
                s += g.pow(2)
                n += (e-g).pow(2)
            var_e += v/nw

        var_e /= gviter
        # Division by gviter cancels out in ss/nn
        snr_e = sum(
                [((ss+1e-10).log()-(nn+1e-10).log()).sum()
                    for ss, nn in zip(Es, En)])/nw
        return Ege, var_e, snr_e

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def _snap_model_gavg(self, model):
        # g_avg = self.opt.g_avg
        # if self.model is None:
        #     if g_avg < 1:
        #         self.model = copy.deepcopy(model)
        #     else:
        #         self.model = model
        #     return
        # for m, g in zip(model.parameters(), self.model.parameters()):
        #     if g_avg < 1:
        #         g.data.mul_(1-self.g_avg).add_(m.data.mul(self.g_avg))
        #     else:
        #         g.data.copy_(m.data)
        pass

    def snap_model(self, model):
        opt = self.opt
        if opt.g_avg == 1:
            if self.model is None:
                self.model = model
            return
        if self.model is None:
            self.model = copy.deepcopy(model)
            # running model sum
            self.model_s = copy.deepcopy(model)
            self.g_avg_iter = 1
            return
        # update sum
        for m, s in zip(model.parameters(), self.model_s.parameters()):
            s.data.add_(m.data)
        self.g_avg_iter += 1
        if self.g_avg_iter != opt.g_avg:
            return
        # update snapshot
        for g, s in zip(self.model.parameters(), self.model_s.parameters()):
            g.data.copy_(s.data).div_(opt.g_avg)
            s.data.fill_(0)
        self.g_avg_iter = 0
