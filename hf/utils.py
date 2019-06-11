import pdb
import time
import gzip

import numpy as np
# from tensorflow.contrib.learn.python.learn.datasets import base

import torch
from torch.autograd import Variable, grad


def copy_model(model):
    copied_model = type(model)()
    # copied_model = type(model)(input_size=784, hidden_size=10, output_size=10)
    for copied_param, orig_param in zip(copied_model.parameters(), model.parameters()):
        copied_param.data.copy_(orig_param.data)
    return copied_model


def copy_params(model, requires_grad=True):
    """Makes a copy of model parameters, wrapped in new Variables.
    """
    ps = []
    for param in model.parameters():
        ps.append(Variable(param.data, requires_grad=requires_grad))
        # ps.append(Variable(param.data.clone(), requires_grad=requires_grad))
    return ps


def rewrap(variables, requires_grad=False, clone=False):
    rewrapped = []
    for v in variables:
        if clone:
            rewrapped.append(Variable(v.data.clone(), requires_grad=requires_grad))
        else:
            rewrapped.append(Variable(v.data, requires_grad=requires_grad))
    return tuple(rewrapped)


def flatten(tensors):
    views = []
    for tensor in tensors:
        views.append(tensor.contiguous().view(-1))
    return torch.cat(views, 0)


def unflatten(flat_tensors, model):
    """Unflattens a flat tensor into a list of reshaped tensors that match the sizes of
    model parameters.
    """
    unflattened = []
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        unflattened.append(flat_tensors[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return unflattened


def new_tensor_seq(tensors, wrap_as_var=True, requires_grad=True):
    """Creates a new sequence of random tensors where each tensor is of the same
    size as the corresponding tensor in the `tensors` input sequence.

    If `tensors` is a single tensor, then this still works, and is equivalent to:
        new_tensor = torch.rand(tensors.size())
    """
    if isinstance(tensors, Variable) or isinstance(tensors, torch.Tensor):
        if wrap_as_var:
            return Variable(torch.rand_like(tensors), requires_grad=requires_grad)
        else:
            return torch.rand_like(tensors)

    new_tensors = []
    for tensor in tensors:
        if wrap_as_var:
            new_tensor = Variable(torch.rand_like(tensor), requires_grad=requires_grad)
        else:
            new_tensor = torch.rand_like(tensor)

        new_tensors.append(new_tensor)

    return new_tensors


##################################
###   Matrix-Vector Products   ###
##################################

def hessian_vec_bk(ys, xs, vs, grads=None):
    """Implements Hessian vector product using backward on backward AD.

    Args:
        ys: Loss function.
        xs: Weights, list of tensors.
        vs: List of tensors to multiply, for each weight tensor.

    Returns:
        Hv: Hessian vector product, same size, same shape as xs.
    """
    if grads is None:
        grads = grad(ys, xs, create_graph=True)
    return grad(grads, xs, vs, create_graph=True)  # Not sure if we need create_graph here


def hessian_vec_fw(ys, xs, vs, grads=None):
    """Implements Hessian vector product using forward on backward AD.

    Args:
        ys: Loss function.
        xs: Weights, list of tensors.
        vs: List of tensors to multiply, for each weight tensor.

    Returns:
        Hv: Hessian vector product, same size, same shape as xs.
    """
    if grads is None:
        grads = grad(ys, xs, create_graph=True)
    return Rop(grads, xs, vs)


def Rop(f_x, x, u):
    """Computes the Jacobian-vector product df(x)dx u
    """
    v = new_tensor_seq(f_x)
    g_v = grad(f_x, x, grad_outputs=v, create_graph=True)  # v^T df(x)/dx  vector-Jacobian product
    vjp_g_u_v = grad(g_v, v, grad_outputs=u)
    return vjp_g_u_v


def fisher_vec_z(ys, xs, vs):
    """Implements JJ'v, where v is on the output space.

    Args:
        ys: Loss function or output variables.
        xs: Weights, list of tensors.
        vs: List of tensors to multiply, for each weight tensor.

    Returns:
        JJ'v: Fisher vector product on the output space.
    """
    jv = grad(ys, xs, grad_outputs=vs, create_graph=True)
    jjv = Rop(ys, xs, jv)
    return jjv


def fisher_vec_bk(ys, xs, vs):
    """Implements Fisher vector product using backward AD.

    Args:
        ys: Loss function, scalar.
        xs: Weights, list of tensors.
        vs: List of tensors to multiply, for each weight tensor.

    Returns:
        J'Jv: Fisher vector product.
    """
    grads = grad(ys, xs, create_graph=True)
    gradsv = [torch.sum(gi * vi) for (gi, vi) in zip(grads, vs)]
    jv = sum(gradsv)  # Does using Python sum work here?
    jjv = list(map(lambda x: x * jv, grads))
    return jjv


def fisher_vec_fw(ys, xs, vs):
    """Implements Fisher vector product using backward and forward AD.

    Args:
        ys: Loss function or output variables.
        xs: Weights, list of tensors.
        vs: List of tensors to multiply, for each weight tensor.

    Returns:
        J'Jv: Fisher vector product.
    """
    jv = Rop(ys, xs, vs)
    jjv = grad(ys, xs, jv, create_graph=True)
    return jjv


def gauss_newton_vec(ys, zs, xs, vs):
    """Implements Gauss-Newton vector product.

    Args:
        ys: Loss function.
        zs: Before output layer (input to softmax).
        xs: Weights, list of tensors.
        vs: List of perturbation vector for each weight tensor.

    Returns:
        J'HJv: Guass-Newton vector product.
    """
    grads_z = grad(ys, zs, create_graph=True)
    hjv = Rop(grads_z, xs, vs)
    jhjv = grad(zs, xs, hjv, create_graph=True)
    return jhjv, hjv

    # This is a slightly different implementation of the same thing
    # grads_z = grad(ys, zs, create_graph=True)
    # Jv = Rop(zs, xs, vs)                            # \ these are in some way "doubled" from the impl. above
    # HJv = grad(grads_z, zs, Jv, create_graph=True)  # /
    # Gv = grad(zs, xs, HJv, create_graph=True)
    # return Gv, HJv


def gauss_newton_vec_z(ys, zs, xs, vs):
    """Implements HJJ'v, where v is on the output space.

    Args:
        ys: Loss function or output variables.
        zs: Before output layer (input to softmax).
        xs: Weights, list of tensors.
        vs: List of tensors to multiply, for each weight tensor.

    Returns:
        HJJ'v: Gauss-Newton vector product on the output space.
    """
    grads_z = grad(ys, zs, create_graph=True)
    jv = grad(zs, xs, vs, create_graph=True)
    hjjv = Rop(grads_z, xs, jv)
    return hjjv


def get_flat_params_from(model):
    """From: https://github.com/Khrylx/PyTorch-RL/blob/master/utils/torch.py

    Example:
        flat_params = get_flat_params_from(model)
        flat_gradient = compute_flat_grad(loss, model.parameters())
        updated_flat_params = flat_params - lr * flat_gradient
        set_flat_params_to(model, updated_flat_params)
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    """From: https://github.com/Khrylx/PyTorch-RL/blob/master/utils/torch.py
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def set_flat_grad_to(model, flat_grad):
    """UNTESTED. Sets the gradient of each model parameter based on flat_grad.

    Example:
        preds = model(inputs)
        loss = criterion(preds, targets)
        flat_gradient = compute_flat_grad(loss, model.parameters)
        set_flat_grad_to(model, flat_gradient)
        optimizer.step()  # Can use this because we've set the param.grad values.
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad = flat_grad[prev_ind:prev_ind + flat_size].view(param.size())
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    """From: https://github.com/Khrylx/PyTorch-RL/blob/master/utils/torch.py
    """
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(Variable(zeros(param.data.view(-1).shape)))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    """From: https://github.com/Khrylx/PyTorch-RL/blob/master/utils/torch.py

    Example:
        preds = model(inputs)
        loss = criterion(preds, targets)
        flat_gradient = compute_flat_grad(loss, model.parameters)
    """
    if create_graph:
        retain_graph = True

    # This part just filters the model parameters based on filter_input_ids (why?)
    # ----------------------------------------------------------------------------
    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)
    # ----------------------------------------------------------------------------

    # Computes the gradient of the output (loss) wrt the model parameters, like normal.
    # The output of this, grads, is a list of tensors (of whatever shape).
    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(Variable(zeros(param.data.view(-1).shape)))
        else:
            out_grads.append(grads[j].view(-1))  # Adds the flattened gradient of one parameter to out_grads
            j += 1
    grads = torch.cat(out_grads)  # Sticks together all flattened grads into one (column?) vector

    for param in params:
        param.grad = None

    return grads



# Time tracking functions
global_time_list = []
global_last_time = 0

def reset_time():
    global global_time_list, global_last_time
    global_time_list = []
    global_last_time = time.perf_counter()


def record_time():
    global global_last_time, global_time_list
    new_time = time.perf_counter()
    global_time_list.append(new_time - global_last_time)
    global_last_time = time.perf_counter()
    #print("step: %.2f"%(global_time_list[-1]*1000))


def summarize_time(time_list=None):
    if time_list is None:
        time_list = global_time_list

    # delete first large interval if exists
    if time_list and time_list[0]>3600*10:
        del time_list[0]

    time_list = 1000*np.array(time_list)  # get seconds, convert to ms
    if len(time_list)>0:
        min = np.min(time_list)
        median = np.median(time_list)
        formatted = ["%.2f"%(d,) for d in time_list[:10]]
        print("Times: min: %.2f, median: %.2f, mean: %.2f"%(min, median,
                                                            np.mean(time_list)))
    else:
        print("Times: <empty>")


def ng_init(s1, s2): # uniform weight init from Ng UFLDL
    r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
    flat = np.random.random(s1*s2)*2*r-r
    return flat.reshape([s1, s2]).astype(np.float32)


def get_mnist_images(fold='train'):
    """Returns mnist images, batch dimension last."""

    def extract_images(f):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
        Args:
          f: A file object that can be passed into a gzip reader.
        Returns:
          data: A 4D uint8 numpy array [index, y, x, depth].
        Raises:
          ValueError: If the bytestream does not start with 2051.
        """
        #    print('Extracting', f.name) # todo: remove
        with gzip.GzipFile(fileobj=f) as bytestream:
          magic = _read32(bytestream)
          if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
          num_images = _read32(bytestream)
          rows = _read32(bytestream)
          cols = _read32(bytestream)
          buf = bytestream.read(rows * cols * num_images)
          data = np.frombuffer(buf, dtype=np.uint8)
          data = data.reshape(num_images, rows, cols, 1)
          return data

    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    if fold == 'train': # todo: rename
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    elif fold == 'test':
        TRAIN_IMAGES = 't10k-images-idx3-ubyte.gz'
    else:
        assert False, 'unknown fold %s'%(fold)

    source_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    local_file = base.maybe_download(TRAIN_IMAGES, '/tmp', source_url + TRAIN_IMAGES)
    train_images = extract_images(open(local_file, 'rb'))
    dsize = train_images.shape[0]
    if fold == 'train':
        assert dsize == 60000
    else:
        assert dsize == 10000

    train_images = train_images.reshape(dsize, 28**2).T.astype(np.float64)/255
    train_images = np.ascontiguousarray(train_images)
    return train_images.astype(np.float32)
