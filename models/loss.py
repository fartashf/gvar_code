import torch
import torch.nn.functional as F

from cusvd import svdj


class Loss(object):
    def __call__(self, model, data,
                 reduction='mean', weights=1, return_output=False):
        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = self.loss(output, target, reduction=reduction)*weights
        if return_output:
            return loss, output
        return loss


class NLLLoss(Loss):
    def __init__(self):
        self.loss = F.nll_loss
        self.do_accuracy = True

    def loss_sample(self, output, target, empirical=False):
        with torch.no_grad():
            if empirical:
                sampled_y = target
            else:
                # probs = torch.nn.functional.softmax(output, dim=1)
                probs = torch.exp(output)  # BUG: models return log_softmax
                sampled_y = torch.multinomial(probs, 1).squeeze()
        loss_sample = self.loss(output, sampled_y, reduction='none')
        return loss_sample

    def fisher(self, output, n_samples):
        S = n_samples
        F_L = 0
        for i in range(S):
            output = output.detach()
            output.requires_grad = True
            probs = torch.exp(output)
            sampled_y = torch.multinomial(probs, 1).squeeze()
            loss_sample = self.loss(output, sampled_y, reduction='none')
            loss_sample.backward()
            F_L += output.grad @ output.grad.T
        return F_L


class MSELoss(Loss):
    def __init__(self):
        # self.loss = F.mse_loss
        self.do_accuracy = False

    def loss(self, *args, **kwargs):
        reduction = 'mean'
        if 'reduction' in kwargs:
            reduction = kwargs['reduction']
        kwargs['reduction'] = 'none'

        ret = F.mse_loss(*args, **kwargs)
        # TODO: what is the prob dist for this loss?
        if reduction == 'mean':
            return ret.sum(1).mean(0)  # ret.mean(1).mean(0)
        return ret.sum(1)  # ret.mean(1)

    def loss_sample(self, output, target, empirical=False):
        with torch.no_grad():
            if empirical:
                sampled_y = target
            else:
                sampled_y = torch.normal(output, torch.ones_like(output))
        loss_sample = self.loss(output, sampled_y, reduction='none')
        return loss_sample

    def fisher(self, output, n_samples=-1):
        if n_samples == -1:
            F_L = torch.eye(output.shape[1], dtype=output.dtype,
                            device=output.device)
            F_L = F_L.unsqueeze(0).expand(output.shape + (output.shape[1],))
            Q_L = F_L
            return F_L, Q_L
        F_L = 0
        output = output.detach()
        output.requires_grad = True
        for i in range(n_samples):
            with torch.no_grad():  # Watch out for other loss functions
                sampled_y = torch.normal(output, torch.ones_like(output))
            loss_sample = 0.5 * self.loss(output, sampled_y, reduction='none')
            loss_sample = loss_sample.sum()
            ograd = torch.autograd.grad(loss_sample, [output])[0]
            F_L += torch.einsum('bo,bp->bop', ograd, ograd)
        F_L /= n_samples
        Q_F = []
        eps = 1e-7
        for i in range(F_L.shape[0]):
            U, S, V = svdj(F_L[i])
            assert all(S+eps > 0), 'S has negative elements'
            Q_F += [U @ (S+eps).sqrt().diag()]
        Q_F = torch.stack(Q_F)
        return F_L, Q_F

    def fisher_fast(self, output, n_samples=-1):
        raise NotImplementedError("")
        output = output.detach()
        output.requires_grad = True
        with torch.no_grad():  # Watch out for other loss functions
            sampled_y = torch.normal(
                output, torch.ones_like(output), n_samples)
        loss_sample = self.loss(
            output.unsqueeze(1), sampled_y, reduction='none')
        loss_sample = loss_sample.sum()
        ograd = torch.autograd.grad(loss_sample, [output])[0]
        F_L = torch.einsum('bo,bp->bop', ograd, ograd)
        F_L /= n_samples
        Q_F = []
        eps = 1e-7
        for i in range(F_L.shape[0]):
            U, S, V = svdj(F_L[i])
            assert all(S+eps > 0), 'S has negative elements'
            Q_F += [U @ (S+eps).sqrt().diag()]
        Q_F = torch.stack(Q_F)
        return F_L, Q_F


class KFACNLL(object):
    def __init__(self):
        self.optim = None

    def __call__(self, model, data, reduction='mean', weights=1):
        optim = self.optim

        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction=reduction)*weights
        if self.optim is not None and optim.steps % optim.TCov == 0:
            # compute true fisher
            optim.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(
                    torch.nn.functional.softmax(output.cpu().data, dim=1),
                    1).squeeze().cuda()
            loss_sample = (F.nll_loss(output, sampled_y,
                                      reduction='none')*weights).mean()
            loss_sample.backward(retain_graph=True)
            optim.acc_stats = False
            model.zero_grad()  # clear the gradient for computing true-fisher.
        return loss
