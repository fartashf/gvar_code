from collections import OrderedDict


def mnist(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_ntk' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
    shared_args = [('dataset', dataset),
                   ('lr', [.1, .05, .01]),
                   ('weight_decay', 0),
                   ('momentum', [0, 0.9]),
                   ('epochs', [
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   ('arch', ['mlp']),  # 'cnn'
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 # ('gvar_log_iter', 100),
                 ('gvar_start', 0),
                 # ('g_bsnap_iter', 1),
                 ('g_optim', ''),
                 ('g_optim_start', 0),
                 ('g_epoch', ''),
                 ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_ntk = [('g_estim', ['ntk']),
                ('ntk_damping', [1e-3, 1e-1])]
    args += [OrderedDict(shared_args+gvar_args+args_ntk)]
    return args, log_dir, module_name, exclude


def cifar10(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_ntk' % dataset
    exclude = ['dataset', 'epochs',
               'g_epoch', 'lr_decay_epoch', 'gvar_log_iter', 'niters']
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   # ('arch', 'resnet20'),
                   # ('arch', 'resnet56'),
                   ('arch', 'resnet8'),
                   # ('epochs', [
                   #     (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                   # ]),
                   # ('niters', 80000),
                   # ('lr_decay_epoch', '40000,60000'),
                   ('epochs', 5),
                   # ('lr', 0.1),
                   ('weight_decay', [0, 1e-4]),
                   ('gvar_log_iter', 200),
                   ('batch_size', 1024),  # [128, 256]),  # [128, 64]),
                   ('lr', [.1]),  # [.1, .05, .02, .01]),
                   ('momentum', 0.9),
                   ]
    gvar_args = [
        # ('gvar_estim_iter', 10),  # default
        # ('gvar_log_iter', 100),  # default
        ('gvar_start', 0),
        # ('g_bsnap_iter', 1),
        ('g_optim', ''),
        ('g_optim_start', 0),  # [0, 10, 20]),
        ('g_epoch', ''),
    ]
    # args_sgd = [('g_estim', ['sgd'])]
    # args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_ntk = [('g_estim', ['ntk']),
                ('ntk_damping', [1e-4, 1e-5]),  # [1e-3, 1e-1]),
                ('ntk_cpu', '')]
    args += [OrderedDict(shared_args+gvar_args+args_ntk)]
    return args, log_dir, module_name, exclude
