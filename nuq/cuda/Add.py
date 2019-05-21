import torch
import math

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
    bucket_size = 16
    asize = a.size()
    num_tail = math.ceil(a.numel()/bucket_size)*bucket_size-a.numel()
    av = torch.cat((a.view(-1), torch.zeros_like(a)[:num_tail]))
    c = torch.zeros_like(a)
    av = av.view(-1, bucket_size)
    norm = av.norm(dim=1, keepdim=True).expand(
        av.shape[0], av.shape[1]).contiguous().view(-1).contiguous()
    print('norm', norm)
    r = torch.randint_like(a, 1000001).long()
    levels = get_uniform_levels(4).cuda()
    print('levels', levels)
    print('#levels', len(levels))
    qdq = QDQ(bucket_size, levels)

    qdq.qdqGPU(a, norm, c, r)
    return c.view(asize)
