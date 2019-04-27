import torch
import ntk.layer


class NeuralTangentKernel(object):
    def __init__(self, model, eps=1e-3, debug=True, **kwargs):
        self.model = model
        self.eps = eps
        self.debug = debug
        self.ntk = ntk.layer.Container(model, debug=debug, **kwargs)

    def activate(self):
        self.ntk.activate()

    def eval(self):
        self.ntk.eval()

    def deactivate(self):
        self.ntk.deactivate()

    def copy_(self, model):
        for m, g in zip(model.parameters(), self.model.parameters()):
            g.data.copy_(m.data)

    def get_kernel(self):
        batch_kernel = self.ntk.get_kernel()
        total_kernel = torch.stack(batch_kernel).sum(0)
        return total_kernel

    def get_kernel_inverse(self):
        with torch.no_grad():
            K = self.get_kernel()
            K += self.eps*torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
            Ki = K.pinverse().detach()
        return Ki
