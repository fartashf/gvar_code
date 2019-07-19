import torch
import torch.nn.functional as F


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
        if reduction == 'mean':
            return ret.mean(1).mean(0)
        return ret.mean(1)

    def loss_sample(self, output, target, empirical=False):
        with torch.no_grad():
            if empirical:
                sampled_y = target
            else:
                sampled_y = torch.normal(output, torch.ones_like(output))
        loss_sample = self.loss(output, sampled_y, reduction='none')
        return loss_sample


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
