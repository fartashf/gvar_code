import pdb
import sys
sys.path.insert(0, '..')  # So that we can import utils

from functools import reduce

import numpy as np

import torch
from torch.autograd import grad
from torch.optim import Optimizer
from torch.autograd import Variable

# Local imports
import utils
from utils import rewrap, flatten, unflatten

import matplotlib.pyplot as plt


plot_num = 0


def cg(f, x, preconditioner=None, lambda_=1, max_iters=10, curvature_matrix_type='G'):
    """When the closure is evaluated, it computes the value of the loss function with the current
    model parameters.
    """
    global plot_num

    phi_list = []

    # To store each step of CG, such that we can backtrack to an earlier CG iteration if we realize
    # that later iterations (that optimize the quadratic approximation well) do not decrease the
    # objective function.
    backtracking = []
    true_losses, quadratic_losses = [], []

    x_orig = [element for element in x]  # Convert x from a generator to a list

    # Testing full batch loss/gradient
    full_batch_loss, full_batch_predictions = f(full_batch=True)
    loss, predictions = f()

    gradient = grad(loss, x_orig, create_graph=True)
    b = rewrap(gradient)
    x = [Variable(torch.zeros(element.size()), requires_grad=True) for element in x_orig]  # Convert x from a generator to a list

    Ax = utils.gauss_newton_vec(loss, predictions, x_orig, x)[0]  # To get J'HJv
    # pdb.set_trace()  # To check that Ax = 0 (when we initialize x to be 0). Checks out!
    Ax_orig = rewrap(Ax)

    # Simplified version of the r and p computations above, using the fact that Ax=0 when the initial x=0
    r = tuple(-item for item in b)
    p = rewrap(b)

    r_k_norm = torch.matmul(flatten(r), flatten(r))

    hfs, hf_fulls = [], []

    for i in range(max_iters):

        Ap = utils.gauss_newton_vec(loss, predictions, x_orig, rewrap(p))[0]
        # Ap_full = utils.gauss_newton_vec(full_batch_loss, full_batch_predictions, x_orig, rewrap(p))[0]  # Temporary, for the plot

        # hfs.append(torch.matmul(flatten(p), flatten(Ap)).item())
        # hf_fulls.append(torch.matmul(flatten(p), flatten(Ap_full)).item())

        # Tikhonov damping
        for (Ap_component, p_component) in zip(Ap, rewrap(p)):
            Ap_component.data = Ap_component.data + lambda_ * p_component.data  # Ap + damping * p

        alpha = r_k_norm / torch.matmul(flatten(p), flatten(Ap))  # checked

        # Update x
        for (x_component, p_component) in zip(x, p):
            x_component.data = x_component.data + alpha.item() * p_component.data  # x + alpha * p

        # Update r
        for (r_component, Ap_component) in zip(r, Ap):
            r_component.data = r_component.data + alpha.item() * Ap_component.data  # r + alpha * Ap

        r_kplus1_norm = torch.matmul(flatten(r), flatten(r))
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm

        for (p_component, r_component) in zip(p, r):
            p_component.data = -r_component.data + beta.item() * p_component.data  # p = -r + beta * p

        current_x_value = tuple(x_component - x_delta for (x_component, x_delta) in zip(x_orig, x))
        current_loss, _ = f(current_x_value)  # Check how well the current parameters do on the true cost function
        backtracking.append((current_loss.item(), rewrap(x)))

        loss_change = current_loss - loss

        neg_x = tuple(-x_delta for x_delta in x)
        Ax_approx = utils.gauss_newton_vec(loss, predictions, x_orig, rewrap(neg_x))[0]
        quadratic_approx_value = loss + torch.matmul(flatten(gradient), flatten(neg_x)) + 0.5 * torch.matmul(flatten(neg_x), flatten(Ax_approx))
        quadratic_losses.append(quadratic_approx_value.item() - loss.item())
        true_losses.append(loss_change.item())

        # compute quadratic
        # phi = 0.5 * np.dot(r - b, x)
        # phi_list.append(phi)
        phi_list.append(quadratic_approx_value.item())
        k = np.minimum(i, 10)

        # if (M(x_j) - M(x_{j-k})) / (M(x_j) - M(0)) < 0.0001:
        # Why do we need to "force" i > 5?
        if i > 5 and (quadratic_approx_value.item() - phi_list[i-k]) / (quadratic_approx_value.item() - loss.item()) < 0.0001:
            break

        # if phi < 0 and i > 10 and (phi - phi_list[i-k]) / phi < k * tol:
        #     break

        # TODO: Need better stopping condition
        if r_k_norm.item() < 0.0001:
            break


    j = np.argmin([b[0] for b in backtracking])
    x = backtracking[j][1]

    # plt.figure()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.scatter(hfs, hf_fulls, s=20)
    # plt.xlabel('hf', fontsize=15)
    # plt.ylabel('hf-full', fontsize=15)
    # plt.tight_layout()

    # xy_line = np.linspace(np.min(hfs), np.max(hfs), 10)
    # plt.plot(xy_line, xy_line, '--k')
    # plt.savefig('hf_current_vs_full.pdf')

    return x, quadratic_losses[-1], i, loss


# def line_search(f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
#     fval = f(x)

#     for stepfrac in [.5**x for x in range(max_backtracks)]:
#         x_new = x + stepfrac * fullstep
#         fval_new = f(x_new)
#         actual_improve = fval - fval_new
#         expected_improve = expected_improve_full * stepfrac
#         ratio = actual_improve / expected_improve

#         if ratio > accept_ratio:
#             return x_new

#     return x


def line_search(closure, params, update, gradient, initial_alpha=1.0, max_backtracks=10):
    params_copy = [Variable(element.data, requires_grad=False) for element in params]  # Make a copy
    initial_value, _ = closure()
    beta = 0.5
    c = 0.01
    alpha = initial_alpha
    # for stepfrac in [beta**i for i in range(max_backtracks)]:
    for _ in range(max_backtracks):
        for param_copy_component, param_component, update_component in zip(params_copy, params, update):
            param_copy_component.data = param_component.data - alpha * update_component.data
        value, _ = closure(params_copy)
        if value.item() <= initial_value.item() + c * alpha * torch.matmul(flatten(gradient), flatten(update)).item():
            return alpha
        else:
            # alpha = initial_alpha * stepfrac
            alpha = alpha * 0.5
    return alpha


# def line_search(f, x, fullstep, grads, c=0.001, max_iter=10):
#     fval = f(x)
#     alpha = 1.
#     for _ in range(max_iter):
#         xnew = x + alpha * fullstep
#         newfval = f(xnew)

#         if newfval <= fval + c * alpha * np.dot(grads, fullstep):
#             return xnew, alpha * fullstep
#         alpha = alpha * 0.5
#     return x + alpha * fullstep, alpha * fullstep


class HfOptimizer(Optimizer):
    """Implements Hessian-Free optimization.

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of conjugate-gradient iterations per optimization step
            (default: 20)
    """

    def __init__(self, model, max_iters=30, curvature_matrix_type='G'):
        """Initializes the HfOptimizer.

        Args:
            model: A model extending the nn.Module class.
            max_iters: The maximum number of iterations of conjugate gradient.
            curvature_matrix_type: Hessian ('H'), Gauss-Newton matrix ('G'), or Fisher matrix ('F')
        """

        defaults = dict(max_iters=max_iters)
        super(HfOptimizer, self).__init__(model.parameters(), defaults)

        self.model = model
        self.lambda_ = 1.0  # Tikhonov damping parameter
        self.max_iters = max_iters
        self.curvature_matrix_type = curvature_matrix_type

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        # Compute pre-conditioning matrix -- expensive for large batch sizes
        # squared_grads = []
        # example_losses, _ = closure(return_example_losses=True)

        # # Compute the gradient of the loss with respect to each example in the batch individually
        # for example_loss in example_losses:
        #     example_grad = grad(example_loss, self.model.parameters(), retain_graph=True)
        #     squared_grads.append(flatten(example_grad) ** 2)

        # mean_squared_grad = sum(squared_grads) / len(squared_grads)
        # preconditioner = (mean_squared_grad + 1) ** -0.75
        # preconditioner = unflatten(preconditioner, self.model)

        # Run conjugate gradient algorithm
        update, q_model_change, iters_run, loss = cg(closure,
                                                     self.model.parameters(),
                                                     preconditioner=None,
                                                     lambda_=self.lambda_,
                                                     max_iters=self.max_iters,
                                                     curvature_matrix_type=self.curvature_matrix_type)

        loss, _ = closure()
        gradient = grad(loss, self.model.parameters(), create_graph=True)  # Should this be the full-batch or mini-batch gradient?

        # alpha = 1
        alpha = line_search(closure, self.model.parameters(), update, gradient, initial_alpha=1.0)
        print("alpha = {}".format(alpha))
        # if alpha < 1:
        #     pdb.set_trace()

        # Update step
        for param_component, update_component in zip(self.model.parameters(), update):
            param_component.data = param_component.data - alpha * update_component.data


        # Levenberg-Marquardt
        # -------------------
        # Now that we've updated the parameters, check the new value of the objective function
        new_loss, _ = closure()
        rho = (new_loss.item() - loss.item()) / q_model_change

        if rho > 0.75:
            self.lambda_ = 0.9 * self.lambda_
            # self.lambda_ = (2.0/3) * self.lambda_
        elif rho < 0.25:
            self.lambda_ = 1.1 * self.lambda_
            # self.lambda_ = (3.0/2) * self.lambda_

        print('rho = {}, lambda_ = {}, loss = {}'.format(rho, self.lambda_, loss.item()))

        return loss
