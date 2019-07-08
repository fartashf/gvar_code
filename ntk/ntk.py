import torch
import ntk.layer
from cusvd import svdj


class NeuralTangentKernel(object):
    def __init__(self, model, damping=1e-3, debug=True, cpu=True,
                 max_sweeps=100, divn=False, sqrt=False, **kwargs):
        self.model = model
        self.damping = damping
        self.debug = debug
        self.ntk = ntk.layer.Container(model, debug=debug, **kwargs)
        self.cpu = cpu
        self.max_sweeps = max_sweeps
        self.divn = divn
        self.sqrt = sqrt

    def activate(self):
        self.ntk.activate()

    def deactivate(self):
        self.ntk.deactivate()

    def get_kernel(self):
        batch_kernel = self.ntk.get_kernel()
        total_kernel = torch.stack(batch_kernel).sum(0)
        if self.divn:
            return total_kernel/total_kernel.shape[0]
        return total_kernel

    def get_kernel_inverse(self):
        with torch.no_grad():
            K = self.get_kernel()
            if not self.cpu:
                ftype = 1
                if ftype == 1:
                    U, S, V = svdj(K, max_sweeps=self.max_sweeps)
                    # self.S = S.clone()
                    if self.sqrt:
                        S = S.sqrt()
                    Si = (1./(S+self.damping))
                    Ki = U @ Si.diag() @ V.t()
                elif ftype == 2:
                    self.Si = Si.clone()
                    K += self.damping*torch.eye(
                        K.shape[0], dtype=K.dtype, device=K.device)
                    Ki = K.inverse().detach()
                elif ftype == 3:
                    import skcuda
                    import skcuda.linalg
                    import pycuda.autoinit  # NOQA
                    import pycuda.gpuarray as gpuarray
                    skcuda.linalg.init()
                    Ki = torch.Tensor(skcuda.linalg.inv(
                        gpuarray.to_gpu(K.cpu().numpy())).get())
                elif ftype == 4:
                    U, S, V = K.svd()
                    Ki = U @ (1./(S+self.damping)).diag() @ V.t()
                elif ftype == 5:
                    S, V = K.symeig(eigenvectors=True)
                    Ki = V @ (1./(S+self.damping)).diag() @ V.t()
                elif ftype == 6:
                    K += self.damping*torch.eye(
                        K.shape[0], dtype=K.dtype, device=K.device)
                    Ki = torch.cholesky_inverse(torch.cholesky(K))
            else:
                ftype = 1
                if ftype == 1:
                    K += self.damping*torch.eye(
                        K.shape[0], dtype=K.dtype, device=K.device)
                    Ki = K.cpu().inverse().detach()
                elif ftype == 2:
                    import numpy as np
                    Ki = torch.Tensor(np.linalg.inv(K.cpu().numpy()))
                elif ftype == 3:
                    import scipy
                    Ki = torch.Tensor(scipy.linalg.inv(K.cpu().numpy()))
                elif ftype == 4:
                    U, S, V = K.cpu().svd()
                    Ki = U @ (1./(S+self.damping)).diag() @ V.t()
            # self.Ki = Ki.clone()
        return Ki.cuda(), S.clone()
