import torch
import torch.nn
import models.mnist
import models.cifar10
import models.logreg
import models.linreg
import models.imagenet
import models.cifar10_wresnet
import models.cifar10_wresnet2
import models.loss
import models.cifar10_vgg
import models.rf


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
    elif (opt.dataset == 'cifar10' or opt.dataset == 'svhn'
          or opt.dataset == 'cifar100'):
        # model = torch.nn.DataParallel(
        #     models.cifar10.__dict__[opt.arch]())
        # model.cuda()
        if opt.arch == 'cnn':
            model = models.cifar10.Convnet(num_class=opt.num_class)
        elif opt.arch == 'scnn':
            model = models.cifar10.SmallCNN(num_class=opt.num_class)
        elif opt.arch == 'sscnn':
            model = models.cifar10.SuperSmallCNN(num_class=opt.num_class)
        elif opt.arch == 'mlp':
            model = models.cifar10.MLP(num_class=opt.num_class)
        elif opt.arch == 'smlp':
            model = models.cifar10.SmallMLP(num_class=opt.num_class)
        elif opt.arch == 'msmlp':
            model = models.cifar10.MoreSmallMLP(num_class=opt.num_class)
        elif opt.arch == 'ssmlp':
            model = models.cifar10.SuperSmallMLP(num_class=opt.num_class)
        elif opt.arch == 'linear':
            model = models.cifar10.LP(num_class=opt.num_class)
        elif opt.arch.startswith('wrn'):
            depth, widen_factor = map(int, opt.arch[3:].split('-'))
            # model = models.cifar10_wresnet.Wide_ResNet(28, 10, 0.3, 10)
            model = models.cifar10_wresnet2.WideResNet(
                depth, opt.num_class, widen_factor, 0.3)
        elif opt.arch.startswith('vgg'):
            model = models.cifar10_vgg.__dict__[opt.arch](
                num_class=opt.num_class)
        else:
            model = models.cifar10.__dict__[opt.arch](
                num_class=opt.num_class, nobatchnorm=opt.nobatchnorm)
        model = torch.nn.DataParallel(model)
    elif opt.dataset == 'imagenet':
        model = models.imagenet.Model(opt.arch, opt.pretrained,
                                      half_trained=opt.half_trained)
    elif opt.dataset.startswith('imagenet'):
        model = models.imagenet.Model(opt.arch, opt.pretrained, opt.num_class)
    elif opt.dataset == 'logreg':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '10class':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '5class':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == 'linreg':
        model = models.linreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == 'rcv1' or opt.dataset == 'covtype':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == 'rf':
        if opt.arch == 'rf':
            model = models.rf.RandomFeaturesModel(
                opt.dim, opt.student_hidden, opt.num_class)
        elif opt.arch == 'resnet32':
            model = models.cifar10.resnet32(
                num_class=opt.num_class, nobatchnorm=opt.nobatchnorm)
            model = torch.nn.DataParallel(model)

    if opt.dataset == 'linreg':
        model.criterion = models.loss.MSELoss()
    elif opt.optim == 'kfac' or opt.optim == 'ekfac':
        model.criterion = models.loss.KFACNLL()
    else:
        model.criterion = models.loss.CELoss()
    if opt.cuda:
        model.cuda()

    return model
