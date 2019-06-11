import numpy as np
import torch


def lanczos_np(A,
               num_iter,
               q0=None,
               re_orthogonalization=True,
               gram_schmidt_step=2,
               dtype=np.float64,
               EPS=1e-12):

    dim = A.shape[0]
    gamma = [None] * num_iter
    beta = [None] * (num_iter + 1)
    beta[-1] = 0.0
    q = [None] * (num_iter + 2)
    q[-1] = np.zeros([dim, 1], dtype=dtype)
    q[0] = np.random.randn(dim, 1).astype(dtype)
    q[0] /= np.sqrt(np.sum(q[0] * q[0]))

    if isinstance(q0, np.ndarray):
        if q0.ndim == 1:
            q[0] = np.expand_dims(q0, axis=1)
        elif q0.ndim == 2:
            q[0] = q0
        else:
            raise ValueError("wrong shape of q0!")

    step = 0
    for ii in range(num_iter):
        z = np.dot(A, q[ii])
        gamma[ii] = np.sum(z * q[ii])
        z = z - gamma[ii] * q[ii] - beta[ii - 1] * q[ii - 1]

        if re_orthogonalization is True:
            for kk in range(gram_schmidt_step):
                for jj in range(ii):
                    z -= np.dot(z.T, q[jj]) * q[jj]

        beta[ii] = np.linalg.norm(z)

        if beta[ii] < EPS:
            break

        q[ii + 1] = z / beta[ii]

        step += 1

    Q = np.stack(q[:step], axis=1)

    return Q, beta[:step - 1], gamma[:step]


# List of Tensors (lot) operations
def lot_uop(op, x):
    return [op(xi) for xi in x]


def lot_pop(op, x, y):
    return [op(xi, yi) for xi, yi in zip(x, y)]


def lot_sop(op, scalar, x):
    return [op(scalar, xi) for xi in x]


def lot_sop2(op, x, scalar):
    return [op(xi, scalar) for xi in x]


def lot_sum(x):
    return torch.sum(torch.cat([xi.flatten() for xi in x]))


def lot_norm(q):
    norm = torch.sqrt(lot_sum(lot_pop(torch.mul, q, q)))
    return norm


def lot_flat(x):
    return torch.cat([xi.flatten() for xi in x])


def lot_unflat(x, params):
    d = 0
    x_unflat = []
    for p in params:
        x_unflat += [x[d:d+p.numel()].view(p.shape)]
        d += p.numel()
    return x_unflat


def lanczos_torch_lot(loss,
                      params,
                      num_iter,
                      q0=None,
                      re_orthogonalization=True,
                      gram_schmidt_step=2,
                      EPS=1e-7,
                      dot_op=torch.dot):
    gamma = [None] * num_iter
    beta = [None] * (num_iter + 1)
    beta[-1] = 0.0
    q = [None] * (num_iter + 2)
    q[-1] = lot_uop(torch.zeros_like, params)
    q[0] = lot_uop(torch.randn_like, params)
    norm = lot_norm(q[0])
    for qi in q[0]:
        qi.div_(norm)

    if isinstance(q0, np.ndarray):
        if q0.ndim == 1:
            q[0] = torch.expand_dims(torch.Tensor(q0), axis=1)
        elif q0.ndim == 2:
            q[0] = q0
        else:
            raise ValueError("wrong shape of q0!")

    step = 0
    for ii in range(num_iter):
        # z = np.dot(A, q[ii])
        z = dot_op(loss, params, q[ii])
        gamma[ii] = lot_uop(torch.sum, lot_pop(torch.mul, z, q[ii]))
        z = lot_pop(torch.sub,
                    lot_pop(torch.sub,
                            z,
                            lot_pop(torch.mul,
                                    gamma[ii],
                                    q[ii])),
                    lot_sop(torch.mul, beta[ii - 1], q[ii - 1]))

        if re_orthogonalization is True:
            for kk in range(gram_schmidt_step):
                for jj in range(ii):
                    z = lot_pop(torch.sub,
                                z,
                                lot_sop(torch.mul,
                                        lot_sum(lot_pop(torch.mul, z, q[jj])),
                                        q[jj]))

        beta[ii] = lot_norm(z)

        if beta[ii] < EPS:
            break

        q[ii + 1] = lot_sop2(torch.div, z, beta[ii])

        step += 1

    Q = torch.stack(q[:step], axis=1)

    return Q, beta[:step - 1], gamma[:step]


def lanczos_torch(loss,
                  params,
                  num_iter,
                  q0=None,
                  re_orthogonalization=True,
                  gram_schmidt_step=2,
                  EPS=1e-7,
                  dot_op=torch.dot):

    params_flat = lot_flat(params)
    gamma = [None] * num_iter
    beta = [None] * (num_iter + 1)
    beta[-1] = 0.0
    q = [None] * (num_iter + 2)
    q[-1] = torch.zeros_like(params_flat)
    if q0 is not None:
        q[0] = torch.clone(lot_flat(q0))
    else:
        q[0] = torch.randn_like(params_flat)
    q[0] /= torch.norm(q[0])

    step = 0
    for ii in range(num_iter):
        qii_unflat = lot_unflat(q[ii], params)
        z_unflat = dot_op(loss, params, qii_unflat)
        z = lot_flat(z_unflat)

        gamma[ii] = torch.sum(z * q[ii])
        z = z - gamma[ii] * q[ii] - beta[ii - 1] * q[ii - 1]

        if re_orthogonalization is True:
            for kk in range(gram_schmidt_step):
                for jj in range(ii):
                    z -= torch.dot(z.t(), q[jj]) * q[jj]

        beta[ii] = torch.norm(z)

        if beta[ii] < EPS:
            break

        q[ii + 1] = z / beta[ii]

        step += 1

    Q = torch.stack(q[:step], dim=1)

    return Q, beta[:step - 1], gamma[:step]
