import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_reset(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)
        m.weight.div_(m.weight.norm(dim=1, keepdim=True))
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight)
        norm = m.weight.view(m.weight.shape[0], -1).norm(dim=1, keepdim=True)
        m.weight.div_(norm.reshape(-1, 1, 1, 1))


class RandomFeaturesModel(nn.Module):
    """
    Random features model in Montanari et al. 2020, Sec. 2.3.1:
    https://arxiv.org/pdf/1911.01544.pdf
    """
    def __init__(self, dim, hidden, num_class):
        super(RandomFeaturesModel, self).__init__()
        self.linear1 = nn.Linear(dim, hidden, bias=False)
        self.linear1.weight.requires_grad = False
        self.linear2 = nn.Linear(hidden, num_class)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            weight_reset(self.linear1.weight)
            weight_reset(self.linear2.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def parameters(self, *args, **kwargs):
        params = super(RandomFeaturesModel, self).parameters(*args, **kwargs)
        for p in params:
            if p is not self.linear1.weight:
                yield p
