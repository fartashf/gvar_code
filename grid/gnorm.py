from collections import OrderedDict


def mnist_gnorm(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gnorm' % dataset
    exclude = ['dataset', 'lr', 'weight_decay', 'epochs', 'lr_decay_epoch',
               'optim', 'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'dim', 'niters']
    shared_args = [('dataset', dataset),
                   ('lr', .02),
                   # ('arch', ['mlp', 'cnn']),
                   ('arch', 'cnn'),
                   # ('weight_decay', 0),
                   ('niters', 50000),
                   ('lr_decay_epoch', 50000),
                   # ('seed', [123, 456, 789]),
                   ('nodropout', ''),
                   ('batch_size', 128),
                   # ## imbalance
                   # ('imbalance', [None, '0,0.01', '0,10', '1,0.01', '1,10']),
                   ]
    gvar_args = [
        ('gvar_estim_iter', 50),  # default 10
        ('gvar_log_iter', 500),  # default 100
        ('optim', 'sgd'),
        ('gvar_start', 10),
        ('g_bsnap_iter', 2000),
    ]

    args_sgd = [
        ('g_estim', ['sgd']),
        ('g_batch_size', [128, 256]),
    ]
    args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    return args, log_dir, module_name, exclude


def cifar10_gnorm(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gnorm' % dataset
    exclude = ['dataset', 'lr', 'weight_decay', 'epochs', 'lr_decay_epoch',
               'optim', 'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'dim', 'niters', 'gvar_log_iter',
               'gvar_estim_iter']
    shared_args = [('dataset', dataset),
                   # ('lr', 0.1),  # .01),  # 0.1
                   ('arch', [
                       # ('cnn', OrderedDict([('nodropout', '')])),
                       ('resnet8',
                        OrderedDict([('nobatchnorm', ''),
                                     ('lr', 0.01)])),
                       # 'resnet32',
                       # 'vgg11'
                       # ('resnet32', OrderedDict([('ghostbn', '')])),
                       # ('vgg11', OrderedDict([('ghostbn', '')])),
                       # 'resnet110',
                   ]),
                   # ('weight_decay', 0),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   # ('seed', [123, 456, 789]),
                   # ## data_prod
                   # ('label_smoothing', [None, 0.1]),
                   # ('data_aug', [None, '']),
                   # ('corrupt_perc', [None, 20]),
                   # ## imbalance
                   # ('imbalance', [None, '0,0.01', '0,10', '1,0.01', '1,10']),
                   # ### duplicate
                   # ('duplicate', '10,10000'),
                   ]
    gvar_args = [
        ('gvar_estim_iter', 50),  # default 10
        ('gvar_log_iter', 500),  # default 100
        ('optim', 'sgd'),
        ('gvar_start', 100),  # 24100),  # 100),
        ('g_bsnap_iter', 20000),
    ]
    disj_args = [  # [],
                 # OrderedDict([('label_smoothing', 0.1)]),
                 # OrderedDict([('duplicate', '10,10000')]),
                 # # OrderedDict([('data_aug', '')]),
                 # # OrderedDict([('max_train_size', 100)]),
                 # OrderedDict([('corrupt_perc', 20)])
                 ]

    args_sgd = [
        ('g_estim', ['sgd']),
        ('g_batch_size', [128, 256]),
    ]
    args += [tuple((OrderedDict(shared_args+args_sgd+gvar_args), disj_args))]

    return args, log_dir, module_name, exclude


def cifar100_gnorm(args):
    dataset = 'cifar100'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gnorm' % dataset
    exclude = ['dataset', 'lr', 'weight_decay', 'epochs', 'lr_decay_epoch',
               'optim', 'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'dim', 'niters', 'gvar_log_iter',
               'gvar_estim_iter']
    shared_args = [('dataset', dataset),
                   ('lr', 0.1),  # .01),  # 0.1
                   ('arch', [
                       # ('cnn', OrderedDict([('nodropout', '')])),
                       # ('resnet8',
                       #  OrderedDict([('nobatchnorm', ''),
                       #               ('lr', 0.01)])),
                       'resnet32',
                       # 'vgg11'
                       # ('resnet32', OrderedDict([('ghostbn', '')])),
                       # ('vgg11', OrderedDict([('ghostbn', '')])),
                   ]),
                   # ('weight_decay', 0),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   # ('seed', [123, 456, 789]),
                   # ('label_smoothing', [None, 0.1]),
                   # ('data_aug', [None, '']),
                   # ### corrupt dup
                   # ('corrupt_perc', [None, 20]),
                   # ('duplicate', '10,10000'),
                   ]
    gvar_args = [
        ('gvar_estim_iter', 50),  # default 10
        ('gvar_log_iter', 500),  # default 100
        ('optim', 'sgd'),
        ('gvar_start', 100),  # 24100),  # 100),
        ('g_bsnap_iter', 20000),
    ]
    disj_args = [  # [],
                 # OrderedDict([('label_smoothing', 0.1)]),
                 # OrderedDict([('duplicate', '10,10000')]),
                 # # OrderedDict([('data_aug', '')]),
                 # # OrderedDict([('max_train_size', 100)]),
                 # OrderedDict([('corrupt_perc', 20)])
                 ]

    args_sgd = [
        ('g_estim', ['sgd']),
        ('g_batch_size', [128, 256]),
    ]
    args += [tuple((OrderedDict(shared_args+args_sgd+gvar_args), disj_args))]

    return args, log_dir, module_name, exclude


def imagenet_gnorm(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gnorm' % dataset
    exclude = ['dataset', 'lr', 'weight_decay', 'epochs', 'lr_decay_epoch',
               'optim', 'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'dim', 'niters', 'gvar_log_iter',
               'gvar_estim_iter']
    shared_args = [('dataset', dataset),
                   ('lr', 0.1),  # .01),  # 0.1
                   ('arch', 'resnet18'),
                   ('weight_decay', 1e-4),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   ('batch_size', 128),
                   # ('seed', [123, 456, 789]),
                   # ('label_smoothing', [None, 0.1]),
                   # ('corrupt_perc', [None, 20]),
                   # ('duplicate', [None, '10,10000']),
                   ]
    gvar_args = [
        ('gvar_estim_iter', 10),  # default 10
        ('gvar_log_iter', 1000),  # default 100
        ('optim', 'sgd'),
        ('gvar_start', 100),  # 24100),  # 100),
        ('g_bsnap_iter', 10000),
    ]
    disj_args = [  # [],
                 # OrderedDict([('label_smoothing', 0.1)]),
                 # OrderedDict([('duplicate', '10,10000')]),
                 # OrderedDict([('duplicate', '10,100000')]),
                 # # OrderedDict([('data_aug', '')]),
                 # # OrderedDict([('max_train_size', 100)]),
                 # OrderedDict([('corrupt_perc', 20)])
                 ]

    args_sgd = [
        ('g_estim', ['sgd']),
        ('g_batch_size', [128, 256]),  # [64, 256]),  # [128, 256]),
    ]
    args += [tuple((OrderedDict(shared_args+args_sgd+gvar_args), disj_args))]

    return args, log_dir, module_name, exclude


def four_datasets(args):
    args, log_dir, module_name, exclude = imagenet_gnorm(args)
    args, log_dir, module_name, exclude = cifar100_gnorm(args)
    args, log_dir, module_name, exclude = mnist_gnorm(args)
    args, log_dir, module_name, exclude = cifar10_gnorm(args)
    log_dir = 'runs_gnorm'
    exclude = ['lr', 'weight_decay', 'epochs', 'lr_decay_epoch',
               'optim', 'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'dim', 'niters']
    return args, log_dir, module_name, exclude
