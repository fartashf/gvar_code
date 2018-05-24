import torch                                                                                                                                                                                                                                                                                                                                                                
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time


class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x): 
        return x

class Layer(nn.Module):
    def __init__(self, in_planes, planes, BN):
        super(Layer, self).__init__()
        self.bn = BN(planes)
        self.conv = nn.Conv2d(in_planes, planes, 3, padding=1, bias=False)
        self.act = nn.ReLU()

    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, BN, n_layers=3):
        super(Net, self).__init__()
        self.in_planes = 3 
        self.layers = self._make_layers(Layer,  64, n_layers, BN) 
        self.linear = nn.Linear(64, 10) 

    def _make_layers(self, block, planes, num_blocks, BN):
        strides = [1] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, BN))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x): 
        out = self.layers(x)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1) 
        return self.linear(out)

def do_epoch(net, criterion, optimizer, lam):
    net.train()
    aggr_loss = count = 0 
    for _ in range(1000):
        inputs = Variable(torch.cuda.FloatTensor(128,3,32,32).normal_(), requires_grad=True)
        targets = Variable(torch.LongTensor(128).random_(0, 10).cuda())
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward(create_graph=(lam > 0)) 

        # gradient penalty
        if lam > 0:
            gpenalty = inputs.grad.view(inputs.size(0), -1).add(1e-5).norm(1, 1).mean()
            (lam * gpenalty).backward()

        optimizer.step()

        count += 1
        aggr_loss += loss.data[0]

    return aggr_loss / count

def main(net, lam):
    net.cuda()
    cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=.001, momentum=0.9)

    for epoch in range(1):
        time_start = time.time()

        loss = do_epoch(net, criterion, optimizer, lam)

        print("epoch %2d loss %.2f time %d" % (epoch, loss, time.time()-time_start))

n_layers = 3 
BN = Identity
print('No BN and no lambda')
main(Net(BN, n_layers), 0.) 

print('No BN and with lambda')
main(Net(BN, n_layers), .001)

BN = nn.BatchNorm2d
print('With BN and no lambda')
main(Net(BN, n_layers), 0.)

print('With BN and with lambda')
main(Net(BN, n_layers), .001)
