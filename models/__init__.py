import torch
import torch.nn
import models.mnist
import models.cifar10
import models.logreg
import models.imagenet
import models.cifar10_wresnet
import models.cifar10_wresnet2


def init_model(opt):
    if opt.dataset == 'mnist':
        if opt.arch == 'cnn':
            model = models.mnist.Convnet(not opt.nodropout)
        elif opt.arch == 'bigcnn':
            model = models.mnist.BigConvnet(not opt.nodropout)
        elif opt.arch == 'mlp':
            model = models.mnist.MLP(not opt.nodropout)
        elif opt.arch == 'smlp':
            model = models.mnist.SmallMLP(not opt.nodropout)
        elif opt.arch == 'ssmlp':
            model = models.mnist.SuperSmallMLP(not opt.nodropout)
        # else:
        #     model = models.mnist.MNISTNet()
    elif opt.dataset == 'cifar10' or opt.dataset == 'svhn':
        # model = torch.nn.DataParallel(
        #     models.cifar10.__dict__[opt.arch]())
        # model.cuda()
        if opt.arch == 'cnn':
            model = models.cifar10.Convnet()
        elif opt.arch.startswith('wrn'):
            depth, widen_factor = map(int, opt.arch[3:].split('-'))
            # model = models.cifar10_wresnet.Wide_ResNet(28, 10, 0.3, 10)
            model = models.cifar10_wresnet2.WideResNet(
                depth, opt.num_class, widen_factor, 0.3)
        else:
            model = models.cifar10.__dict__[opt.arch]()
        model = torch.nn.DataParallel(model)
    elif opt.dataset == 'imagenet':
        model = models.imagenet.Model(opt.arch, opt.pretrained)
    elif opt.dataset.startswith('imagenet'):
        model = models.imagenet.Model(opt.arch, opt.pretrained, opt.num_class)
    elif opt.dataset == 'logreg':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '10class':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '5class':
        model = models.logreg.Linear(opt.dim, opt.num_class)

    if opt.cuda:
        model.cuda()

    return model
