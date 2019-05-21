import torch
import MakePytorchPlusPlus as MPP
import numpy as np


def test_foo():
    foo = MPP.Foo()
    print(foo)
    foo.setKey(3)
    print(foo)
    print(foo.getKey())


def test_add_gpu():
    if not torch.cuda.is_available():
        return
    a = torch.cuda.FloatTensor(4)
    b = torch.cuda.FloatTensor(4)
    a.normal_()
    b.normal_()
    c = MPP.add_gpu(a, b)
    print(a, b, c)


def test_qdq_gpu():
    if not torch.cuda.is_available():
        return
    x = torch.randn(1000).cuda().uniform_(-1, 1)
    q = MPP.qdq_gpu(x)
    dq = np.unique(q.cpu().numpy())
    print('x', x)
    print('q', q)
    print('unique q', dq)
    print('# unique q', len(dq))


if __name__ == '__main__':
    test_foo()
    # test_add_gpu()
    test_qdq_gpu()
