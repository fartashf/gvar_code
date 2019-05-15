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
    log_dir = 'runs_%s_gvar_ntk_bs1024_sgd' % dataset
    exclude = ['dataset', 'epochs',
               'g_epoch', 'lr_decay_epoch', 'gvar_log_iter', 'niters']
    shared_args = [('dataset', dataset),
                   ('arch', 'resnet32'),
                   # ('arch', 'resnet20'),
                   # ('arch', 'resnet56'),
                   # ('arch', 'resnet8'),
                   # ('epochs', [
                   #     (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                   # ]),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   # ('epochs', 5),
                   # ('lr', 0.1),
                   ('weight_decay', 0),  # [0, 1e-4]),
                   ('gvar_log_iter', 200),  # 200),
                   ('gvar_estim_iter', 5),  # 200),
                   ('batch_size', 1024),  # 1024, 128),
                   # ('lr', [.1]),  # [.1, .05, .02, .01]),
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
    args_sgd = [('g_estim', ['sgd']),
                ('lr', [.12, .15, .2]),  # [.1]),  # [.1, .05, .02, .01]),
                ]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_kfac = [('g_estim', ['sgd']),
                 ('optim', ['kfac']),
                 # ('lr', 0.02),
                 # ('kf_damping',  0.03),
                 ('lr', [.1, .05, .02, .01]),
                 ('kf_damping',  [0.05, 0.03, 0.01]),
                 ]
    args += [OrderedDict(shared_args+gvar_args+args_kfac)]

    # args_ntk = [('g_estim', ['ntk']),
    #             ('lr', [5e-3, 1e-2, 2e-2, 5e-2]),  # , 5e-4]),, 2e-3, 1e-3
    #             ('ntk_damping', [5e-2, 3e-2, 1e-2]),  # , 5e-3, 2e-3, 1e-3]),
    #             # 1e-3, 1e-5, 1e-1]),  # [1e-3, 1e-1]),
    #             # ('ntk_cpu', ''),
    #             ]
    # args += [OrderedDict(shared_args+gvar_args+args_ntk)]
    return args, log_dir, module_name, exclude


def mnist_vae(args):
    dataset = 'mnist'
    module_name = 'main.vae'
    log_dir = 'runs_%s_ntk_vae' % dataset
    exclude = ['dataset', 'epochs',
               'g_epoch', 'lr_decay_epoch', 'gvar_log_iter', 'niters']
    shared_args = [('dataset', dataset),
                   ('weight_decay', 0),
                   ('momentum', 0.9),
                   ('epochs', [
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
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
    args_sgd = [('g_estim', ['sgd']),
                ('optim', [
                    ('sgd', OrderedDict([('lr', [.01, 5e-3, 1e-3, 5e-4])])),
                    ('adam', OrderedDict([('lr', [.01, 5e-3, 1e-3, 5e-4])])),
                ]),
                ]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_ntk = [('g_estim', ['ntk']),
                ('lr', [.1, .05, .02, .01]),
                ('ntk_damping', [1e-3, 1e-5, 1e-1]),  # [1e-3, 1e-1]),
                ('ntk_cpu', '')]
    args += [OrderedDict(shared_args+gvar_args+args_ntk)]
    return args, log_dir, module_name, exclude
