import torch
import ntk.layer
# import numpy as np
# import pycuda.autoinit  # NOQA
# import pycuda.gpuarray as gpuarray
# import skcuda
# import skcuda.linalg
# import sys
# sys.path.append('/u/faghri/torch-batch-svd/')
# from batch_svd import svdj  # batch_svd  # NOQA
from cusvd import svdj


class NeuralTangentKernel(object):
    def __init__(self, model, damping=1e-3, debug=True, cpu=True,
                 max_sweeps=100, divn=False, **kwargs):
        self.model = model
        self.damping = damping
        self.debug = debug
        self.ntk = ntk.layer.Container(model, debug=debug, **kwargs)
        self.cpu = cpu
        self.max_sweeps = max_sweeps
        self.divn = divn

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
                U, S, V = svdj(K, max_sweeps=self.max_sweeps)
                # self.S = S.clone()
                Si = (1./(S+self.damping))
                Ki = U @ Si.diag() @ V.t()
                # self.Si = Si.clone()
                # K += self.damping*torch.eye(
                #     K.shape[0], dtype=K.dtype, device=K.device)
                # Ki = K.inverse().detach()
                # skcuda.linalg.init()
                # Ki = torch.Tensor(skcuda.linalg.inv(
                #     gpuarray.to_gpu(K.cpu().numpy())).get())
                # U, S, V = K.svd()
                # Ki = U @ (1./(S+self.damping)).diag() @ V.t()
                # S, V = K.symeig(eigenvectors=True)
                # Ki = V @ (1./(S+self.damping)).diag() @ V.t()
                # K += self.damping*torch.eye(
                #     K.shape[0], dtype=K.dtype, device=K.device)
                # Ki = torch.cholesky_inverse(torch.cholesky(K))
            else:
                K += self.damping*torch.eye(
                    K.shape[0], dtype=K.dtype, device=K.device)
                Ki = K.cpu().inverse().detach()
                # Ki = torch.Tensor(np.linalg.inv(K.cpu().numpy()))
                # Ki = torch.Tensor(scipy.linalg.inv(K.cpu().numpy()))
                # U, S, V = K.cpu().svd()
                # Ki = U @ (1./(S+self.damping)).diag() @ V.t()
            # self.Ki = Ki.clone()
        return Ki.cuda(), S.clone()
