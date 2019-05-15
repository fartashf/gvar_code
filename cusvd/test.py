# https://github.com/KinglittleQ/torch-batch-svd
import torch
from cusvd import svdj


def test():
    torch.manual_seed(0)
    a = torch.randn(9, 3).cuda()
    b = a.clone()

    U, S, V = svdj(a, is_sort=True)

    u, s, v = torch.svd(b, some=True, compute_uv=True)

    print(U.shape)
    print(u.shape)
    print((U.abs() - u.abs()).sum())
    print(torch.allclose(U.abs(), u.abs()))
    print(torch.allclose(S.abs(), s.abs()))
    print(torch.allclose(V.abs(), v.abs()))

    print(a)
    print(b)
    print(U @ S.diag() @ V.t())
    print(u @ s.diag() @ v.t())


if __name__ == '__main__':
    test()

    print('Finished')
