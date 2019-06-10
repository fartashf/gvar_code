def np_Lanczos(A,
               num_iter,
               q0=None,
               re_orthogonalization=True,
               gram_schmidt_step=2,
               dtype=np.float64):

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
  for ii in xrange(num_iter):
    z = np.dot(A, q[ii])
    gamma[ii] = np.sum(z * q[ii])
    z = z - gamma[ii] * q[ii] - beta[ii - 1] * q[ii - 1]

    if re_orthogonalization == True:
      for kk in xrange(gram_schmidt_step):
        for jj in xrange(ii):
          z -= np.dot(z.T, q[jj]) * q[jj]

    beta[ii] = np.linalg.norm(z)

    if beta[ii] < EPS:
      break

    q[ii + 1] = z / beta[ii]

    step += 1

  Q = np.stack(q[:step], axis=1)

  return Q, beta[:step - 1], gamma[:step]