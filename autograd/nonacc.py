import torch
import torch.nn
from torch.autograd import Variable


def nonacc_grad(model, loss):
    grads = []
    params = [x for x in model.parameters()]
    for l in loss:
        model.zero_grad()
        tmp_grads = torch.autograd.grad(l, params, retain_graph=True)
        grads.append(tmp_grads)

    # import ipdb; ipdb.set_trace()
    return [grads]


def nonacc_grad_backward(model, loss):
    # https://discuss.pytorch.org/t/quickly-get-individual-gradients-not-sum-of-gradients-of-all-network-outputs/8405
    # https://github.com/tensorflow/tensorflow/issues/4897#issuecomment-253056458
    output_gradients = []

    for output in loss:
        tmp_grad = []
        model.zero_grad()
        output.backward(retain_graph=True)
        for param in model.parameters():
            tmp_grad.append(param.grad.clone())
        output_gradients.append(tmp_grad)

    return [output_gradients]


def acc_grad(model, grads):
    grads = grads[0]
    for i, param in enumerate(model.parameters()):
        param.grad = grads[0][i].clone()
        for j in range(1, len(grads)):
            param.grad += grads[j][i]
        param.grad /= len(grads)


def nonacc_grad_fast(model, loss):
    # http://pytorch.org/docs/master/_modules/torch/autograd.html#grad
    raise NotImplemented()
    params = [x[1] for x in model.named_parameters()]
    grads = []
    for param in params:
        data = param.data
        grads.append(Variable(torch.ones((loss.numel(),)+data.shape),
                              volatile=True).cuda())

    torch.autograd.grad(loss, params, grads)
    return grads
