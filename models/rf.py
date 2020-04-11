import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.init.normal_(self.linear1.weight)
            self.linear1.weight.div_(
                self.linear1.weight.norm(dim=1, keepdim=True))
            nn.init.normal_(self.linear2.weight)
            self.linear2.weight.div_(
                self.linear2.weight.norm(dim=1, keepdim=True))

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def parameters(self, *args, **kwargs):
        params = super(RandomFeaturesModel, self).parameters(*args, **kwargs)
        for p in params:
            if p is not self.linear1.weight:
                yield p
