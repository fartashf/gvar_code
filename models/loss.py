import torch
import torch.nn.functional as F


def nll_loss(model, data, reduction='mean', weights=1):
    data, target = data[0].cuda(), data[1].cuda()
    model.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction=reduction)*weights
    return loss


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


class TransformerLossKFAC(object):
    def __init__(self):
        self.optim = None

    def __call__(self, model, data, reduction='mean', weights=1):
        optim = self.optim
        data, target = data[0].cuda(), data[1].cuda()

        model.zero_grad()
        mems = tuple()
        ret = model(data, target, *mems)
        loss, pred_hid, mems = ret[0], ret[1], ret[2:]
        loss = loss.mean(0)*weights
        if reduction == 'mean':
            loss = loss.mean()

        if self.optim is not None and optim.steps % optim.TCov == 0:
            # compute true fisher
            optim.acc_stats = True
            loss_sample = (model.loss_sample(pred_hid).mean(0)*weights).mean()
            loss_sample.backward(retain_graph=True)
            optim.acc_stats = False
            model.zero_grad()  # clear the gradient for computing true-fisher.
        return loss
