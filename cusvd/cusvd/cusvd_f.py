import torch  # NOQA
import cusvd_c


def svdj(x, is_sort=False, tol=1e-7, max_sweeps=100):
    U, S, V = cusvd_c.svdj_forward(x, is_sort, tol, max_sweeps)
    # k = S.shape[0]
    # U = U[:, :k]
    # V = V[:, :k]
    return U, S, V
