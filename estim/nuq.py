import torch
import torch.nn
import torch.multiprocessing

from args import opt_to_nuq_kwargs
from .gestim import GradientEstimator
from nuq.quantize import QuantizeSingleBucket


class NUQEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(NUQEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.qdq = QuantizeSingleBucket(**opt_to_nuq_kwargs(self.opt))
        self.ngpu = self.opt.nuq_ngpu

    def grad(self, model_new, in_place=False):
        model = model_new

        acc_grad = []
        model.zero_grad()
        with torch.no_grad():
            for p in model.parameters():
                if in_place:
                    acc_grad += [p.grad]
                else:
                    acc_grad += [torch.zeros_like(p.grad)]

        for i in range(self.ngpu):
            model.zero_grad()
            data = next(self.data_iter)

            loss = model.criterion(model, data)
            grad = torch.autograd.grad(loss, model.parameters())

            with torch.no_grad():
                for g, a in zip(grad, acc_grad):
                    a += self.qdq.quantize(g.view(-1)).view_as(g)
                    # NUMPY
                    # x = g.view(-1).cpu().numpy()
                    # xq = self.qdq.quantize(x)
                    # a += torch.as_tensor(
                    #     xq/self.ngpu,
                    #     dtype=p.dtype, device=p.device).reshape_as(g)

        if in_place:
            return loss
        return acc_grad
