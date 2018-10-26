import torch
import torch.nn as nn
import copy
from torch.nn.parallel.parallel_apply import parallel_apply


class ParallelAutograd(nn.Module):
    def __init__(self, module, batch_size):
        super(ParallelAutograd, self).__init__()
        self.replicas = [module]
        # self.add_module('0', module)
        self.batch_size = batch_size
        for i in range(1, batch_size):
            self.replicas += [copy.deepcopy(module)]
        for i in range(batch_size):
            self.add_module('%d' % i, self.replicas[i])

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        for i in range(1, self.batch_size):
            self.replicas[i].load_state_dict(self.replicas[0].state_dict())
        outputs = parallel_apply(self.replicas, inputs, kwargs)
        return self.gather(outputs)

    def scatter(self, inputs, kwargs):
        x = inputs[0]
        xs = [[x1] for x1 in torch.split(x, 1)]
        kwargs = None
        return xs, kwargs

    def gather(self, outputs):
        return torch.cat(outputs)

    # def cuda(self):
    #     super(ParallelAutograd, self).cuda()
    #     for i in range(self.batch_size):
    #         self.replicas[i].cuda()

    def trainable_parameters(self, *args, **kwargs):
        return self.replicas[0].parameters(*args, **kwargs)


class DataMomentumHook(object):
    def __init__(self, train_size):
        self.data_momentum = torch.ones(train_size).cuda()
        self.grad_norm = torch.ones((train_size,)).cuda()

    def __call__(self, module, grad_input, grad_output):
        for i in range(module.batch_size):
            pass
