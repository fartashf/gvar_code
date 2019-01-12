import torch.nn.functional as F


def nll_loss(model, data, reduction='mean'):
    data, target = data[0].cuda(), data[1].cuda()
    model.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction=reduction)
    return loss
