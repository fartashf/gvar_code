import torch
import torch.nn
import torch.multiprocessing


class GradientEstimator(object):
    def __init__(self, data_loader, opt, tb_logger=None, *args, **kwargs):
        self.opt = opt
        self.model = None
        self.data_loader = data_loader
        self.tb_logger = tb_logger

    def snap_batch(self, model, niters):
        pass

    def update_sampler(self):
        pass

    def snap_online(self, model, niters):
        pass

    def grad(self, model_new, in_place=False):
        raise NotImplemented('grad not implemented')

    def get_Ege_var(self, model, gviter):
        # estimate grad mean and variance
        Ege = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad(model)
            for e, g in zip(Ege, ge):
                e += g

        for e in Ege:
            e /= gviter
        nw = sum([w.numel() for w in model.parameters()])
        var_e = 0
        Es = [torch.zeros_like(g) for g in model.parameters()]
        En = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad(model)
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
