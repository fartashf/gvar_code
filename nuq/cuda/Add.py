import torch

from MakePytorchBackend import AddGPU, QDQ


def add_gpu(a, b):
    assert isinstance(a, torch.cuda.FloatTensor) \
        and isinstance(b, torch.cuda.FloatTensor)
    assert a.numel() == b.numel()

    c = a.new()
    AddGPU(a, b, c)
    return c


def get_uniform_levels(bits):
    num_levels = 2 << bits - 1
    levels_uni = torch.linspace(-1, 1, steps=num_levels)
    return levels_uni


def qdq_gpu(a):
    assert isinstance(a, torch.cuda.FloatTensor)
    bucket_size = 1024
    c = torch.zeros_like(a).cuda()
    levels = get_uniform_levels(4).cuda()
    qdq = QDQ(bucket_size, levels)

    qdq.qdqGPU(a, float((a*a).sum().sqrt()), c)
    return c
