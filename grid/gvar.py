from collections import OrderedDict


def mnist_gvar(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar' % dataset
    exclude = ['dataset', 'arch', 'epochs', 'lr_decay_epoch', 'weight_decay',
               'g_epoch']
    # epoch_iters = 468
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   ('lr', .05),  # .02
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', 100)])),
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('arch', ['mlp', 'cnn']),
                   ('arch', ['cnn']),
                   # ('arch', 'bigcnn', 'ssmlp'),
                   ('optim', ['sgd']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   # ('log_stats', ''),
                   ]
    gvar_args = [
                  # ('gvar_estim_iter', 10),
                  # ('gvar_log_iter', 100),
                  ('gvar_start', 0),
                  ('g_bsnap_iter', 2),
                  ('g_epoch', '')
                  ]
    args_sgd = [('g_estim', ['sgd']),
                ('g_batch_size', [128, 256])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+gvar_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [2, 10, 100]),
            ('g_debug', ''),
            # ('g_CZ', '')
            # ('g_noMulNk', ''),
            ]

    # args_3 = [('gb_citers', 2),
    #           ('g_min_size', 100),
    #           ('gvar_start', 6*epoch_iters),
    #           ('g_bsnap_iter', 6*epoch_iters),
    #           ('g_optim', ''),
    #           ('g_optim_start', 6*epoch_iters+1),
    #           ]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .99),  # 1-lr (the desired learning rate)
              ('g_min_size', .01),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ('g_init_mul', 2),
              ('g_reinit_iter', 1),
              ]
    args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def imagenet_pretrained_gvar(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar' % dataset
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   # ('arch', 'resnet50'),
                   ('batch_size', 128),
                   # ('test_batch_size', 64),
                   #  ### pretrained
                   ('pretrained', ['']),
                   ('epochs', [10]),
                   ('lr', [.001]),
                   # ('lr_decay_epoch', [10]),
                   # ('exp_lr', [None]),
                   ]
    shared_args += [('gvar_estim_iter', 10),  # 100
                    ('gvar_log_iter', 100),
                    ('gvar_start', 1000),
                    ('g_bsnap_iter', 10000)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [10, 100]),
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10)]
    # args += [OrderedDict(shared_args+snap_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .9),  # 1-lr (the desired learning rate)
              ('g_min_size', 1),  # roughly .1*batch_size/nclusters
              # ('g_reinit', 'largest')  # default
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def cifar10_gvar_resnet32(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar' % dataset
    epoch_iters = 390
    shared_args = [('dataset', dataset),
                   ('arch', 'resnet32'),
                   ('epochs', [
                       (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                       # (100, OrderedDict([('lr_decay_epoch', '50,75')])),
                       # (200, OrderedDict([('exp_lr', ''),
                       #                    ('lr_decay_epoch', 100)])),
                       # (100, OrderedDict([('exp_lr', ''),
                       #                    ('lr_decay_epoch', 50)])),
                   ]),
                   ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   ]
    shared_args += [
                    # ('gvar_estim_iter', 10),  # default
                    # ('gvar_log_iter', 100),  # default
                    # ('gvar_start', 10*epoch_iters),
                    # ('g_bsnap_iter', 10*epoch_iters)
                    ('gvar_start', 10*epoch_iters),
                    ('g_bsnap_iter', 10*epoch_iters)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [10, 100]),  # 2
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10),
    #           ('g_min_size', 100)]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .9),  # 1-lr (the desired learning rate)
              ('g_min_size', .01),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def cifar10_gvar_cnn(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_cnn_adam' % dataset
    epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   # ('epochs', [
                   #     (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                   #     # (100, OrderedDict([('lr_decay_epoch', '50,75')])),
                   #     # (200, OrderedDict([('exp_lr', ''),
                   #     #                    ('lr_decay_epoch', 100)])),
                   #     # (100, OrderedDict([('exp_lr', ''),
                   #     #                    ('lr_decay_epoch', 50)])),
                   # ]),
                   # ('lr', 0.1),
                   # ('weight_decay', 1e-4),
                   ('arch', 'cnn'),
                   ('lr', [0.01]),
                   ('momentum', [0.9]),
                   ('optim', ['adam']),
                   ]
    shared_args += [
                    # ('gvar_estim_iter', 10),  # default
                    # ('gvar_log_iter', 100),  # default
                    # ('gvar_start', 10*epoch_iters),
                    # ('g_bsnap_iter', 10*epoch_iters)
                    ('gvar_start', 10*epoch_iters),
                    ('g_bsnap_iter', 1*epoch_iters)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [10, 100]),  # 2
            # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10),
    #           ('g_min_size', 100)]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),  # [5, 10]),
              ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
              ('g_min_size', .1),  # [.1, .01]),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def mnist_gvar_dup(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_dup' % dataset
    epoch_iters = 468
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   ('lr', .02),
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', 100)])),
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   # ('exp_lr', ''),
                   ('arch', ['cnn']),  # 'mlp'
                   # ('arch', 'bigcnn', 'ssmlp'),
                   ('optim', ['sgd']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   # ('log_stats', ''),
                   ]
    shared_args += [
                    # ('gvar_estim_iter', 10),
                    # ('gvar_log_iter', 100),
                    ('gvar_start', 2*epoch_iters),
                    ('g_bsnap_iter', 2*epoch_iters)
                    ]
    disj_args = [[],
                 OrderedDict([('wnoise', '')]),
                 OrderedDict([('label_smoothing', 0.1)]),
                 OrderedDict([('duplicate', '10,10000')]),
                 OrderedDict([('corrupt_perc', [20, 50, 100])])]
    args_sgd = [('g_estim', ['sgd'])]
    args += [tuple((OrderedDict(shared_args+args_sgd), disj_args))]

    args_svrg = [('g_estim', ['svrg'])]
    args += [tuple((OrderedDict(shared_args+args_svrg), disj_args))]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [100]),
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10),
    #           ('g_min_size', 100)]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .99),  # 1-lr (the desired learning rate)
              ('g_min_size', .1),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ]
    args += [tuple((OrderedDict(shared_args+gluster_args+args_4), disj_args))]
    return args, log_dir, module_name, []


def mnist_gvar_bs(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_bs' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   # ('lr', .02),
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', 100)])),
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   # ('exp_lr', ''),
                   ('arch', ['cnn']),  # , 'mlp'
                   # ('arch', 'bigcnn', 'ssmlp'),
                   # ('optim', ['sgd']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   # ('log_stats', ''),
                   ('batch_size', [
                       (8, OrderedDict([
                           ('lr', .005),
                           ('gvar_estim_iter', 100),
                           ('gvar_log_iter', 1000),
                           ('gvar_start', 2*(60000//8)),
                           ('g_bsnap_iter', 2*(60000//8))
                           ])),
                       # (128, OrderedDict([('lr', .02)])),
                       (1024, OrderedDict([
                           ('lr', .05),
                           ('gvar_estim_iter', 10),
                           ('gvar_log_iter', 10),
                           ('gvar_start', 2*(60000//1024)),
                           ('g_bsnap_iter', 2*(60000//1024))
                           ])),
                   ]),
                   ]
    shared_args += [
                    # ('gvar_estim_iter', 10),
                    # ('gvar_log_iter', 100),
                    # ('gvar_start', 2*epoch_iters),
                    # ('g_bsnap_iter', 2*epoch_iters)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [2, 10, 100]),
            ('g_debug', ''),
            # ('g_CZ', '')
            # ('g_noMulNk', ''),
            ]

    # args_3 = [('gb_citers', 10),
    #           ('g_min_size', 100)]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .99),  # 1-lr (the desired learning rate)
              ('g_min_size', .01),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def cifar10_gvar_cnn_bs(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_cnn_bs' % dataset
    # epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   # ('epochs', [
                   #     (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                   #     # (100, OrderedDict([('lr_decay_epoch', '50,75')])),
                   #     # (200, OrderedDict([('exp_lr', ''),
                   #     #                    ('lr_decay_epoch', 100)])),
                   #     # (100, OrderedDict([('exp_lr', ''),
                   #     #                    ('lr_decay_epoch', 50)])),
                   # ]),
                   # ('lr', 0.1),
                   # ('weight_decay', 1e-4),
                   ('arch', 'cnn'),
                   # ('lr', [0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   ('batch_size', [
                       (32, OrderedDict([
                           ('lr', .005),
                           ('gvar_estim_iter', 100),
                           ('gvar_log_iter', 1000),
                           ('gvar_start', 10*(50000//32)),
                           ('g_bsnap_iter', (50000//32))
                           ])),
                       # (128, OrderedDict([('lr', .01)])),
                       (1024, OrderedDict([
                           ('lr', .05),
                           ('gvar_estim_iter', 10),
                           ('gvar_log_iter', 10),
                           ('gvar_start', 10*(50000//1024)),
                           ('g_bsnap_iter', 1*(50000//1024))
                           ])),
                   ]),
                   ]
    shared_args += [
                    # ('gvar_estim_iter', 10),  # default
                    # ('gvar_log_iter', 100),  # default
                    # ('gvar_start', 10*epoch_iters),
                    # ('g_bsnap_iter', 10*epoch_iters)
                    # ('gvar_start', 10*epoch_iters),
                    # ('g_bsnap_iter', 1*epoch_iters)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [2, 10, 100]),  # 2
            # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10),
    #           ('g_min_size', 100)]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),  # [5, 10]),
              ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
              ('g_min_size', .001),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def svhn_gvar(args):
    # Same as cifar10 resnet32
    dataset = 'svhn'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar' % dataset
    epoch_iters = 604388//128
    shared_args = [('dataset', dataset),
                   ('arch', 'resnet32'),
                   ('epochs', [
                       (160, OrderedDict([('lr_decay_epoch', '80,120')])),
                       # (120, OrderedDict([('lr_decay_epoch', '60,90')])),
                       # (160, OrderedDict([('exp_lr', ''),
                       #                    ('lr_decay_epoch', 80)])),
                   ]),
                   # ('batch_size', [
                   #     (128, OrderedDict([('lr', .1)])),
                   #     (1024, OrderedDict([('lr', .5)])),
                   # ]),
                   ('weight_decay', 1e-4),
                   # ('lr', 0.1),
                   # ('log_nex', ''),
                   # ('wnoise', [None, '']),
                   ]
    shared_args += [
                    # ('gvar_estim_iter', 10),  # default
                    # ('gvar_log_iter', 100),  # default
                    # ('gvar_start', 10*epoch_iters),
                    # ('g_bsnap_iter', 10*epoch_iters)
                    ('gvar_start', 10*epoch_iters),
                    ('g_bsnap_iter', 1*epoch_iters)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [2, 10, 100]),  # 2
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10),
    #           ('g_min_size', 100)]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),  # [5, 10]),
              ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
              ('g_min_size', .001),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def imagenet_gvar(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar' % dataset
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   # ('arch', 'resnet50'),
                   ('batch_size', 128),
                   # ('test_batch_size', 64),
                   #  ### pretrained
                   # ('pretrained', ['']),
                   # ('epochs', [10]),
                   # ('lr', [.001]),
                   # ('lr_decay_epoch', [10]),
                   # ('exp_lr', [None]),
                   ]
    shared_args += [('gvar_estim_iter', 10),  # 100
                    ('gvar_log_iter', 100),
                    ('gvar_start', 1000),
                    ('g_bsnap_iter', 10000)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [10, 100]),
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10)]
    # args += [OrderedDict(shared_args+snap_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .9),  # 1-lr (the desired learning rate)
              ('g_min_size', 1),  # roughly .1*batch_size/nclusters
              # ('g_reinit', 'largest')  # default
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def mnist_gvar_optim(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_optim' % dataset
    # epoch_iters = 60000//128
    exclude = ['dataset', 'arch', 'epochs', 'lr_decay_epoch', 'weight_decay',
               'g_epoch']
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   # ('lr', .02),
                   # ('lr', [.1, .02, .001]),
                   # ('lr', [.1, .05]),
                   ('lr', [.05]),
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', 100)])),
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   # ('exp_lr', ''),
                   ('arch', ['cnn']),  # , 'mlp'
                   # ('arch', 'bigcnn', 'ssmlp'),
                   # ('optim', ['sgd']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   # ('log_stats', ''),
                   # ('wnoise', ''),
                   # ('wnoise_stddev', 1.),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),
                 # ('gvar_log_iter', 100),
                 ('gvar_start', 0),
                 ('g_bsnap_iter', 1),
                 ('g_optim', ''),
                 ('g_optim_start', 2),
                 ('g_epoch', ''),
                 ]
    args_sgd = [('g_estim', ['sgd']),
                ('g_batch_size', [128, 256])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+gvar_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [2, 10, 100]),
            ('g_debug', ''),
            # ('g_CZ', '')
            # ('g_noMulNk', ''),
            # ('g_noise', [0.1, 0.01, 0.001]),
            ]

    # args_3 = [('gb_citers', 2),
    #           ('g_min_size', 100),
    #           ('gvar_start', 6),
    #           ('g_bsnap_iter', 6),
    #           ('g_optim', ''),
    #           ('g_optim_start', 6),
    #           ]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .99),  # 1-lr (the desired learning rate)
              ('g_min_size', .01),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ('g_init_mul', 2),
              ('g_reinit_iter', 1),
              ]
    args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def cifar10_gvar_cnn_optim(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_cnn_optim' % dataset
    epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   # ('epochs', [
                   #     (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                   #     # (100, OrderedDict([('lr_decay_epoch', '50,75')])),
                   #     # (200, OrderedDict([('exp_lr', ''),
                   #     #                    ('lr_decay_epoch', 100)])),
                   #     # (100, OrderedDict([('exp_lr', ''),
                   #     #                    ('lr_decay_epoch', 50)])),
                   # ]),
                   # ('lr', 0.1),
                   # ('weight_decay', 1e-4),
                   ('arch', 'cnn'),
                   # ('lr', [0.1, 0.01, 0.001]),
                   ('lr', [0.1, 0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   ]
    gvar_args = [
                 # ('gvar_estim_iter', 10),  # default
                 # ('gvar_log_iter', 100),  # default
                 # ('gvar_start', 10*epoch_iters),
                 # ('g_bsnap_iter', 10*epoch_iters)
                 ('gvar_start', 10*epoch_iters),
                 ('g_bsnap_iter', 1*epoch_iters),
                 ('g_optim', ''),
                 ('g_optim_start', 10*epoch_iters+1),
                 ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+gvar_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [2, 10, 100]),  # 2
            # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
            ('g_debug', '')]

    args_3 = [('gb_citers', 2),
              ('g_min_size', 100),
              ('gvar_start', 10*epoch_iters),
              ('g_bsnap_iter', 8*epoch_iters),
              ('g_optim', ''),
              ('g_optim_start', 10*epoch_iters+1),
              ]
    args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),  # [5, 10]),
              ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
              ('g_min_size', .001),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ('g_init_mul', 2)
              ]
    args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, []


def cifar10_gvar_resnet_bs_optim(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_resnet_bs_optim' % dataset
    exclude = ['dataset', 'arch', 'epochs', 'lr_decay_epoch', 'weight_decay',
               'g_epoch']
    # epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   ('arch', 'resnet20'),
                   ('epochs', [
                       (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                       # (100, OrderedDict([('lr_decay_epoch', '50,75')])),
                       # (200, OrderedDict([('exp_lr', ''),
                       #                    ('lr_decay_epoch', 100)])),
                       # (100, OrderedDict([('exp_lr', ''),
                       #                    ('lr_decay_epoch', 50)])),
                   ]),
                   # ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   # ('arch', 'cnn'),
                   # ('lr', [0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   ('batch_size', [
                       # (32, OrderedDict([
                       #     ('lr', .005),
                       #     ('gvar_estim_iter', 10),
                       #     ('gvar_log_iter', 1000),
                       #     ('gvar_start', 10*(50000//32)),
                       #     ('g_bsnap_iter', (50000//32))
                       #     ])),
                       (32, OrderedDict([
                           # ('lr', [.05, .02, .01]),
                           ('lr', [.05]),
                           ('gvar_estim_iter', 10),
                           ('gvar_log_iter', 2000),
                           ])),
                       (64, OrderedDict([
                           # ('lr', [.1, .05, .02]),
                           ('lr', [.05]),
                           ('gvar_estim_iter', 10),
                           ('gvar_log_iter', 2000),
                           ])),
                       # (128, OrderedDict([
                       #     ('lr', [.2, .1, .05]),
                       #     ('gvar_estim_iter', 10),
                       #     ('gvar_log_iter', 1000),
                       #     ])),
                       # (128, OrderedDict([('lr', .01)])),
                       # (1024, OrderedDict([
                       #     ('lr', .05),
                       #     ('gvar_estim_iter', 10),
                       #     ('gvar_log_iter', 10),
                       #     ('gvar_start', 10*(50000//1024)),
                       #     ('g_bsnap_iter', 1*(50000//1024))
                       #     ])),
                   ]),
                   ]
    gvar_args = [
                  # ('gvar_estim_iter', 10),  # default
                  # ('gvar_log_iter', 100),  # default
                  # ('gvar_start', 101),
                  # ('g_bsnap_iter', 1),
                  # ('g_optim', ''),
                  # ('g_optim_start', 101),
                  # ('g_epoch', ''),
                  ('gvar_start', 2),
                  ('g_bsnap_iter', 2),
                  ('g_optim', ''),
                  ('g_optim_start', 2),
                  ('g_epoch', ''),
                  ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    # args_svrg = [('g_estim', ['svrg'])]
    # args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [32, 64]),  # [10, 100]),  # 2
            # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
            ('g_debug', ''),
            # ('g_optim_max', 50)
            ]

    args_3 = [('gb_citers', 5),
              ('g_min_size', 100),
              ('gvar_start', 2),
              ('g_bsnap_iter', 10),  # citers*4*bsnap_svrg
              ('g_optim', ''),
              ('g_optim_start', 2),
              ('g_epoch', ''),
              ]
    args += [OrderedDict(shared_args+gluster_args+args_3)]
    # args_4 = [('g_online', ''),
    #           ('g_osnap_iter', 10),  # [5, 10]),
    #           ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
    #           ('g_min_size', .001),  # 100x diff in probabilities
    #           # ('g_reinit', 'largest')
    #           ('g_init_mul', 2)
    #           ]
    # args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def imagenet_gvar_half(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_half' % dataset
    exclude = ['dataset', 'arch', 'weight_decay', 'half_trained']
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   # ('arch', 'resnet50'),
                   ('batch_size', 128),
                   # ('test_batch_size', 64),
                   ('weight_decay', 1e-4),
                   #  ### pretrained
                   ('half_trained', ['']),
                   ('epochs', [45]),
                   ('lr', [.01]),
                   ('lr_decay_epoch', ['15,45']),
                   # ('exp_lr', [None]),
                   ]
    shared_args += [('gvar_estim_iter', 10),  # 100
                    ('gvar_log_iter', 1000),
                    ('gvar_start', 10000),
                    ('g_bsnap_iter', 10000)
                    ]
    args_sgd = [('g_estim', ['sgd'])]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg'])]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', [10, 100]),
            ('g_debug', '')]

    # args_3 = [('gb_citers', 10)]
    # args += [OrderedDict(shared_args+snap_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .99),  # 1-lr (the desired learning rate)
              ('g_min_size', .01),  # roughly .1*batch_size/nclusters
              # ('g_reinit', 'largest')  # default
              ('g_reinit_iter', 50),
              ('g_init_mul', 2)
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def cifar10_blup_inactive(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_blup_inactive' % dataset
    exclude = ['dataset', 'arch', 'epochs', 'lr_decay_epoch', 'weight_decay',
               'g_epoch', 'resume', 'ckpt_name']
    # epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   ('arch', 'resnet20'),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 50)])),
                   ]),
                   # ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   # ('arch', 'cnn'),
                   # ('lr', [0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   # ('batch_size', [
                   #     # (128, OrderedDict([
                   #     #     ('lr', [.2, .1, .05]),
                   #     #     ('gvar_estim_iter', 10),
                   #     #     ('gvar_log_iter', 1000),
                   #     #     ])),
                   #     (128, OrderedDict([('lr', .01)])),
                   # ]),
                   ('resume',
                    'runs/runs_cifar10_blup/g_estim_gluster,epoch_100,120'),
                   ('ckpt_name', 'model_best.pth.tar'),
                   ('gvar_log_iter', 500),
                   ]
    # gvar_args = [
    #               # ('gvar_estim_iter', 10),  # default
    #               # ('gvar_log_iter', 100),  # default
    #               # ('gvar_start', 101),
    #               # ('g_bsnap_iter', 1),
    #               # ('g_optim', ''),
    #               # ('g_optim_start', 101),
    #               # ('g_epoch', ''),
    #               ('gvar_start', 0),
    #               ('g_bsnap_iter', 1),
    #               ('g_optim', ''),
    #               ('g_optim_start', 0),
    #               ('g_epoch', ''),
    #               ]
    # args_sgd = [('g_estim', ['sgd']),
    #             ('lr', [0.01]),
    #             ]
    # args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    # args_svrg = [('g_estim', ['svrg']),
    #              ('lr', [0.01, .001]),
    #              ]
    # args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('g_nclusters', 100),  # [10, 100]),  # 2
            # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
            ('g_debug', ''),
            # ('g_optim_max', 50)
            ('g_active_only', ['layer3', 'layer2', 'layer1']),
            ('lr', [0.01]),
            ('g_avg', [.9]),
            ]

    args_3 = [('gb_citers', 2),
              ('g_min_size', 100),
              ('gvar_start', 0),
              ('g_bsnap_iter', 1),  # citers*4*bsnap_svrg
              ('g_optim', ''),
              ('g_optim_start', 5),
              ('g_epoch', ''),
              # ('g_stable', [1, 1e4]),  # [1000, 10]),
              ]
    args += [OrderedDict(shared_args+gluster_args+args_3)]
    # args_4 = [('g_online', ''),
    #           ('g_osnap_iter', 10),  # [5, 10]),
    #           ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
    #           ('g_min_size', .001),  # 100x diff in probabilities
    #           # ('g_reinit', 'largest')
    #           ('g_init_mul', 2)
    #           ]
    # args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def cifar10_gvar_resnet_adam(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_resnet_adam_bs' % dataset
    exclude = ['dataset', 'arch', 'epochs', 'lr_decay_epoch', 'weight_decay',
               'g_epoch', 'resume', 'ckpt_name',
               'gvar_log_iter']
    # epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   ('arch', 'resnet20'),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 50)])),
                   ]),
                   # ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   # ('arch', 'cnn'),
                   # ('lr', [0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   # ('batch_size', [
                   #     # (128, OrderedDict([
                   #     #     ('lr', [.2, .1, .05]),
                   #     #     ('gvar_estim_iter', 10),
                   #     #     ('gvar_log_iter', 1000),
                   #     #     ])),
                   #     (128, OrderedDict([('lr', .01)])),
                   # ]),
                   ('resume',
                    'runs/runs_cifar10_blup/g_estim_gluster,epoch_100,120'),
                   ('ckpt_name', 'model_best.pth.tar'),
                   ('gvar_log_iter', 500),
                   ('batch_size', [128, 256]),  # [128, 64]),
                   ('optim', [
                       # ('sgd', OrderedDict([('lr', 0.01)])),
                       ('adam', OrderedDict(
                           [('lr', [1e-3, 5e-4, 2e-4, 1e-4])])),
                       # ('adamw', OrderedDict(
                       #     [('lr', [1e-3, 5e-4, 1e-4, 5e-5])])),
                       ]),
                   ]
    gvar_args = [
                  # ('gvar_estim_iter', 10),  # default
                  # ('gvar_log_iter', 100),  # default
                  # ('gvar_start', 101),
                  # ('g_bsnap_iter', 1),
                  # ('g_optim', ''),
                  # ('g_optim_start', 101),
                  # ('g_epoch', ''),
                  ('gvar_start', 0),
                  ('g_bsnap_iter', 2),
                  ('g_optim', ''),
                  ('g_optim_start', 2),
                  ('g_epoch', ''),
                  ]
    args_sgd = [('g_estim', ['sgd']),
                # ('lr', [0.01]),
                # ('optim', [
                #     # ('sgd', OrderedDict([('lr', 0.01)])),
                #     # ('adam', OrderedDict([('lr', [1e-3, 1e-4])])),
                #     ('adamw', OrderedDict([('lr', [1e-3, 1e-4])])),
                #     ]),
                ]
    args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    # args_svrg = [('g_estim', ['svrg']),
    #              # ('lr', [0.01, .001]),
    #              # ('g_avg', [1, 10]),
    #              ('g_msnap_iter', [10]),
    #              # ('optim', 'adamw'),
    #              # ('lr', [1e-4, 1e-4]),  # 1e-3
    #              ]
    # args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    # gluster_args = [
    #         ('g_estim', 'gluster'),
    #         ('g_nclusters', 64),  # [10, 100]),  # 2
    #         # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
    #         ('g_debug', ''),
    #         # ('g_optim_max', 50)
    #         # ('g_active_only', ['layer3', 'layer2', 'layer1']),
    #         # ('lr', [0.01]),
    #         # ('g_avg', [1, 10]),
    #         ('g_msnap_iter', [10]),
    #         # ('optim', 'adamw'),
    #         # ('lr', [1e-3, 5e-4]),
    #         ]

    # # args_3 = [('gb_citers', 2),
    # #           ('g_min_size', 100),
    # #           ('gvar_start', 0),
    # #           ('g_bsnap_iter', 1),  # citers*4*bsnap_svrg
    # #           ('g_optim', ''),
    # #           ('g_optim_start', 5),
    # #           ('g_epoch', ''),
    # #           # ('g_stable', [1, 1e4]),  # [1000, 10]),
    # #           ]
    # # args += [OrderedDict(shared_args+gluster_args+args_3)]
    # args_4 = [('g_online', ''),
    #           ('g_osnap_iter', 10),  # [5, 10]),
    #           ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
    #           ('g_min_size', .001),  # 100x diff in probabilities
    #           # ('g_reinit', 'largest')
    #           ('g_init_mul', 2),
    #           ('g_reinit_iter', 10),
    #           ]
    # args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def cifar10_gvar_resnet_adam_svrg(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_resnet_adam_svrg_2' % dataset
    exclude = ['dataset', 'arch', 'epochs', 'lr_decay_epoch', 'weight_decay',
               'g_epoch', 'resume', 'ckpt_name',
               'gvar_log_iter']
    # epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   ('arch', 'resnet20'),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 50)])),
                   ]),
                   # ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   # ('arch', 'cnn'),
                   # ('lr', [0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   # ('batch_size', [
                   #     # (128, OrderedDict([
                   #     #     ('lr', [.2, .1, .05]),
                   #     #     ('gvar_estim_iter', 10),
                   #     #     ('gvar_log_iter', 1000),
                   #     #     ])),
                   #     (128, OrderedDict([('lr', .01)])),
                   # ]),
                   ('resume',
                    'runs/runs_cifar10_blup/g_estim_gluster,epoch_100,120'),
                   ('ckpt_name', 'model_best.pth.tar'),
                   ('gvar_log_iter', 200),
                   # ('batch_size', [128, 256]),  # [128, 64]),
                   ]
    gvar_args = [
                  # ('gvar_estim_iter', 10),  # default
                  # ('gvar_log_iter', 100),  # default
                  # ('gvar_start', 101),
                  # ('g_bsnap_iter', 1),
                  # ('g_optim', ''),
                  # ('g_optim_start', 101),
                  # ('g_epoch', ''),
                  ('gvar_start', 0),
                  ('g_bsnap_iter', 2),
                  ('g_optim', ''),
                  ('g_optim_start', 2),
                  ('g_epoch', ''),
                  ]
    # args_sgd = [('g_estim', ['sgd']),
    #             ('batch_size', [128, 64]),  # [128, 256]),  # [128, 64]),
    #             # ('lr', [0.01]),
    #             # ('lr', [1e-3, 5e-4, 1e-4]),
    #             # ('optim', ['sgd', 'adam']),
    #             ('optim', [
    #                 ('sgd', OrderedDict([('lr', [1e-2, 1e-3, 5e-4])])),
    #                 ('adam', OrderedDict([('lr', [1e-3, 5e-4, 1e-4])])),
    #                 # ('adamw', OrderedDict([('lr', [1e-3, 1e-4])])),
    #                 ]),
    #             ]
    # args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    args_svrg = [('g_estim', ['svrg']),
                 ('batch_size', 64),
                 ('lr', [4e-4, 3e-4, 2e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5]),
                 # [1e-3, 5e-4, 1e-4, 1e]),
                 # ('lr', [0.01, .001]),
                 # ('g_avg', [1, 10]),
                 ('g_msnap_iter', [10]),
                 # ('optim', 'adamw'),
                 # ('lr', [1e-4, 1e-4]),  # 1e-3
                 ('optim', [
                     # ('sgd', OrderedDict([('lr', 0.01)])),
                     ('adam', OrderedDict([
                          # [('lr', [1e-4, 1e-3]),
                          # [1e-1, 1e-2, 1e-3]),2e-3, 5e-3
                          # [1e-3, 5e-4, 2e-4, 1e-4]),
                          # ('adam_betas',
                          #     # ["'(.9,.9995)'",
                          #     # "'(.9,.999)'", "'(.9,.997)'",
                          #     #  "'(.9,.99)'", "'(.5,.95)'"]),
                          #     ["'(.5,.999)'"]),
                          # ('adam_eps', [1e-8, 1e-3, .1]),
                          # [1e-2, 1e-3, 1e-4]),
                          ])),
                     # ('adamw', OrderedDict(
                     #     [('lr', [1e-3, 5e-4, 1e-4, 5e-5])])),
                     ]),
                 ]
    args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    # gluster_args = [
    #         ('g_estim', 'gluster'),
    #         ('g_nclusters', 64),  # [10, 100]),  # 2
    #         # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
    #         ('g_debug', ''),
    #         # ('g_optim_max', 50)
    #         # ('g_active_only', ['layer3', 'layer2', 'layer1']),
    #         # ('lr', [0.01]),
    #         # ('g_avg', [1, 10]),
    #         ('g_msnap_iter', [10]),
    #         # ('optim', 'adamw'),
    #         # ('lr', [1e-3, 5e-4]),
    #         ]

    # # args_3 = [('gb_citers', 2),
    # #           ('g_min_size', 100),
    # #           ('gvar_start', 0),
    # #           ('g_bsnap_iter', 1),  # citers*4*bsnap_svrg
    # #           ('g_optim', ''),
    # #           ('g_optim_start', 5),
    # #           ('g_epoch', ''),
    # #           # ('g_stable', [1, 1e4]),  # [1000, 10]),
    # #           ]
    # # args += [OrderedDict(shared_args+gluster_args+args_3)]
    # args_4 = [('g_online', ''),
    #           ('g_osnap_iter', 10),  # [5, 10]),
    #           ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
    #           ('g_min_size', .001),  # 100x diff in probabilities
    #           # ('g_reinit', 'largest')
    #           ('g_init_mul', 2),
    #           ('g_reinit_iter', 10),
    #           ]
    # args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def cifar10_gvar_adam_svrg_epoch0(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_resnet_adam_svrg_scratch_train_adamw' % dataset
    exclude = ['dataset', 'arch', 'epochs', 'weight_decay',
               'g_epoch', 'resume', 'ckpt_name', 'lr_decay_epoch',
               'gvar_log_iter', 'gvar_start', 'g_debug', 'g_bsnap_iter']
    # epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   ('arch', 'resnet20'),
                   # ('epochs', 200),
                   ('epochs', [
                       (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                   ]),
                   # ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   # ('arch', 'cnn'),
                   # ('lr', [0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   # ('batch_size', [
                   #     # (128, OrderedDict([
                   #     #     ('lr', [.2, .1, .05]),
                   #     #     ('gvar_estim_iter', 10),
                   #     #     ('gvar_log_iter', 1000),
                   #     #     ])),
                   #     (128, OrderedDict([('lr', .01)])),
                   # ]),
                   # ('resume',
                   #  'runs/runs_cifar10_blup/g_estim_gluster,epoch_100,120'),
                   # ('ckpt_name', 'model_best.pth.tar'),
                   ('gvar_log_iter', 200),
                   # ('batch_size', [128, 256]),  # [128, 64]),
                   ]
    gvar_args = [
                  # ('gvar_estim_iter', 10),  # default
                  # ('gvar_log_iter', 100),  # default
                  # ('gvar_start', 101),
                  # ('g_bsnap_iter', 1),
                  # ('g_optim', ''),
                  # ('g_optim_start', 101),
                  ('g_epoch', ''),
                  ('gvar_start', 0),
                  ('g_bsnap_iter', 2),
                  # ('g_optim_start', [5, 10, 20]),
                  ]
    args_sgd = [('g_estim', ['sgd']),
                ('batch_size', [128, 64]),  # [128, 256]),  # [128, 64]),
                # ('lr', [0.01]),
                # ('lr', [1e-3, 5e-4, 1e-4]),
                # ('optim', ['sgd', 'adam']),
                ('optim', [
                    # ('sgd', OrderedDict([('lr', [.1, .01])])),
                    # ('adam', OrderedDict([('lr', [.01, 1e-3])])),
                    ('adamw', OrderedDict([('lr', [.01, 1e-3])])),
                    ]),
                ]
    args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    args_svrg = [('g_estim', ['svrg']),
                 ('batch_size', 64),
                 ('lr', [1e-2, 1e-3]),  # 1e-4
                 ('g_optim', ''),
                 ('g_optim_start', [20]),  # 5, 10, 20]),
                 ('g_mlr', [1, 2]),  # 0.1, 10]),
                 ('optim', ['adamw']),  # 'adam'
                 # ('g_optim_start', [
                 #     5, 10, 20,
                 #     (5, OrderedDict([('lr_decay_epoch', '5,100,150')])),
                 #     (10, OrderedDict([('lr_decay_epoch', '10,100,150')])),
                 #     (20, OrderedDict([('lr_decay_epoch', '20,100,150')])),
                 #     ]),
                 # ('lr', [0.01, 5e-3, 1e-3, 5e-4, 1e-4]),
                 # ('lr', [4e-4, 3e-4, 2e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5]),
                 # [1e-3, 5e-4, 1e-4, 1e]),
                 # ('lr', [0.01, .001]),
                 # ('g_avg', [1, 10]),
                 # ('g_msnap_iter', [10]),
                 # ('optim', 'adamw'),
                 # ('lr', [1e-4, 1e-4]),  # 1e-3
                 # ('optim', [
                 #     # ('sgd', OrderedDict([('lr', 0.01)])),
                 #     ('adam', OrderedDict([
                 #          # [('lr', [1e-4, 1e-3]),
                 #          # [1e-1, 1e-2, 1e-3]),2e-3, 5e-3
                 #          # [1e-3, 5e-4, 2e-4, 1e-4]),
                 #          # ('adam_betas',
                 #          #     # ["'(.9,.9995)'",
                 #          #     # "'(.9,.999)'", "'(.9,.997)'",
                 #          #     #  "'(.9,.99)'", "'(.5,.95)'"]),
                 #          #     ["'(.5,.999)'"]),
                 #          # ('adam_eps', [1e-8, 1e-3, .1]),
                 #          # [1e-2, 1e-3, 1e-4]),
                 #          ])),
                 #     # ('adamw', OrderedDict(
                 #     #     [('lr', [1e-3, 5e-4, 1e-4, 5e-5])])),
                 #     ]),
                 ]
    args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    # Frequent snaps
    # args_svrg = [('g_estim', ['svrg']),
    #              ('batch_size', 64),
    #              ('lr', 1e-3),  # [1e-2, 1e-3]),  # 1e-4
    #              ('g_optim', ''),
    #              # ('g_optim_start', [5, 10, 20]),
    #              ('g_optim_start', 20*390),
    #              # ('g_mlr', [1, 0.1, 10]),
    #              ('g_mlr', [1, 2, 5]),
    #              ('gvar_start', 20*390),
    #              ('g_bsnap_iter', [50, 100]),
    #              ('optim', 'adam')
    #              ]
    # args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [
            ('g_estim', 'gluster'),
            ('batch_size', 64),
            ('g_nclusters', 64),  # [10, 100]),  # 2
            # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
            ('g_debug', ''),
            # ('g_optim_max', 50)
            # ('g_active_only', ['layer3', 'layer2', 'layer1']),
            # ('lr', [0.01]),
            # ('g_avg', [1, 10]),
            # ('g_msnap_iter', [10]),
            # ('optim', 'adamw'),
            # ('lr', [1e-3, 5e-4]),
            ('lr', [1e-2, 1e-3]),  # 1e-4
            ('optim', 'adamw'),  # 'adam'
            ('g_optim', ''),
            ('g_optim_start', [5]),  # , 10, 20]),
            ('g_mlr', [1, 2]),  # , 5]),
            ]

    # args_3 = [('gb_citers', 2),
    #           ('g_min_size', 100),
    #           ('gvar_start', 0),
    #           ('g_bsnap_iter', 1),  # citers*4*bsnap_svrg
    #           ('g_optim', ''),
    #           ('g_optim_start', 5),
    #           ('g_epoch', ''),
    #           # ('g_stable', [1, 1e4]),  # [1000, 10]),
    #           ]
    # args += [OrderedDict(shared_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),  # [5, 10]),
              ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
              ('g_min_size', .001),  # 100x diff in probabilities
              # ('g_reinit', 'largest')
              ('g_init_mul', 2),
              ('g_reinit_iter', 10),
              ]
    args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def cifar10_gvar_adam_svrg_freq_snap(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_resnet_adam_svrg_freq_snap' % dataset
    exclude = ['dataset', 'epochs', 'weight_decay',
               'g_epoch', 'resume', 'ckpt_name', 'lr_decay_epoch',
               'gvar_log_iter', 'gvar_start', 'g_debug', 'g_bsnap_iter',
               'niters']
    # epoch_iters = 390
    shared_args = [('dataset', dataset),
                   # ('arch', 'resnet32'),
                   # ('arch', 'resnet20'),
                   # ('arch', 'resnet56'),
                   ('arch', 'resnet8'),
                   # ('epochs', 200),
                   # ('epochs', [
                   #     (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                   # ]),
                   # ('niters', 80000),
                   # ('lr_decay_epoch', '40000,60000'),
                   ('niters', 20000),
                   ('lr_decay_epoch', '10000,15000'),
                   ('epoch_iters', 100),
                   # ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   # ('arch', 'cnn'),
                   # ('lr', [0.01]),
                   # ('momentum', [0.9]),
                   # ('optim', ['adam']),
                   # ('batch_size', [
                   #     # (128, OrderedDict([
                   #     #     ('lr', [.2, .1, .05]),
                   #     #     ('gvar_estim_iter', 10),
                   #     #     ('gvar_log_iter', 1000),
                   #     #     ])),
                   #     (128, OrderedDict([('lr', .01)])),
                   # ]),
                   # ('resume',
                   #  'runs/runs_cifar10_blup/g_estim_gluster,epoch_100,120'),
                   # ('ckpt_name', 'model_best.pth.tar'),
                   ('gvar_log_iter', 200),
                   # ('batch_size', [128, 256]),  # [128, 64]),
                   ]
    gvar_args = [
                  # ('gvar_estim_iter', 10),  # default
                  # ('gvar_log_iter', 100),  # default
                  # ('gvar_start', 101),
                  # ('g_bsnap_iter', 1),
                  # ('g_optim', ''),
                  # ('g_optim_start', 101),
                  ('g_epoch', ''),
                  ('gvar_start', 0),
                  ('g_bsnap_iter', 2),
                  # ('g_optim_start', [5, 10, 20]),
                  ]
    args_sgd = [('g_estim', ['sgd']),
                # ('batch_size', [128, 64]),  # [128, 256]),  # [128, 64]),
                ('batch_size', [2**10, 2**11]),  # [256, 512]),
                # ('lr', [0.01]),
                # ('lr', [1e-3, 5e-4, 1e-4]),
                # ('optim', ['sgd', 'adam']),
                ('optim', [
                    ('sgd', OrderedDict([('lr', [.1, .2])])),
                    ('adam', OrderedDict([('lr', [1e-3, 2e-3])])),
                    # ('adamw', OrderedDict([('lr', [.01, 1e-3])])),
                    ]),
                ]
    args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    # args_svrg = [('g_estim', ['svrg']),
    #              ('batch_size', 64),
    #              ('lr', [1e-2, 1e-3]),  # 1e-4
    #              ('g_optim', ''),
    #              ('g_optim_start', [20]),  # 5, 10, 20]),
    #              ('g_mlr', [1, 2]),  # 0.1, 10]),
    #              ('optim', ['adamw']),  # 'adam'
    #              # ('g_optim_start', [
    #              #     5, 10, 20,
    #              #     (5, OrderedDict([('lr_decay_epoch', '5,100,150')])),
    #              #     (10, OrderedDict([('lr_decay_epoch', '10,100,150')])),
    #              #     (20, OrderedDict([('lr_decay_epoch', '20,100,150')])),
    #              #     ]),
    #              # ('lr', [0.01, 5e-3, 1e-3, 5e-4, 1e-4]),
    #              # ('lr', [4e-4, 3e-4, 2e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5]),
    #              # [1e-3, 5e-4, 1e-4, 1e]),
    #              # ('lr', [0.01, .001]),
    #              # ('g_avg', [1, 10]),
    #              # ('g_msnap_iter', [10]),
    #              # ('optim', 'adamw'),
    #              # ('lr', [1e-4, 1e-4]),  # 1e-3
    #              # ('optim', [
    #              #     # ('sgd', OrderedDict([('lr', 0.01)])),
    #              #     ('adam', OrderedDict([
    #              #          # [('lr', [1e-4, 1e-3]),
    #              #          # [1e-1, 1e-2, 1e-3]),2e-3, 5e-3
    #              #          # [1e-3, 5e-4, 2e-4, 1e-4]),
    #              #          # ('adam_betas',
    #              #          #     # ["'(.9,.9995)'",
    #              #          #     # "'(.9,.999)'", "'(.9,.997)'",
    #              #          #     #  "'(.9,.99)'", "'(.5,.95)'"]),
    #              #          #     ["'(.5,.999)'"]),
    #              #          # ('adam_eps', [1e-8, 1e-3, .1]),
    #              #          # [1e-2, 1e-3, 1e-4]),
    #              #          ])),
    #              #     # ('adamw', OrderedDict(
    #              #     #     [('lr', [1e-3, 5e-4, 1e-4, 5e-5])])),
    #              #     ]),
    #              ]
    # args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    # # Frequent snaps
    # args_svrg = [('g_estim', ['svrg']),
    #              # ('batch_size', 64),
    #              ('batch_size', 256),
    #              ('lr', 1e-3),  # [1e-2, 1e-3]),  # 1e-4
    #              ('g_optim', ''),
    #              # ('g_optim_start', [5, 10, 20]),
    #              ('g_optim_start', 20*100),  # 20*390),
    #              # ('g_mlr', [1, 0.1, 10]),
    #              # ('g_mlr', [1, 2, 5]),
    #              ('gvar_start', 20*100),  # 20*390),
    #              ('g_bsnap_iter', [10]),  # [50]),
    #              ('svrg_bsnap_num', [256*5, 256*10, 256*20, 256*40]),
    #              ('optim', 'adam')
    #              ]
    # args += [OrderedDict(shared_args+args_svrg)]

    # gluster_args = [
    #         ('g_estim', 'gluster'),
    #         ('batch_size', 64),
    #         ('g_nclusters', 64),  # [10, 100]),  # 2
    #         # ('g_active_only', ['module.fc2', 'module.fc1,module.fc2']),
    #         ('g_debug', ''),
    #         # ('g_optim_max', 50)
    #         # ('g_active_only', ['layer3', 'layer2', 'layer1']),
    #         # ('lr', [0.01]),
    #         # ('g_avg', [1, 10]),
    #         # ('g_msnap_iter', [10]),
    #         # ('optim', 'adamw'),
    #         # ('lr', [1e-3, 5e-4]),
    #         ('lr', [1e-2, 1e-3]),  # 1e-4
    #         ('optim', 'adamw'),  # 'adam'
    #         ('g_optim', ''),
    #         ('g_optim_start', [5]),  # , 10, 20]),
    #         ('g_mlr', [1, 2]),  # , 5]),
    #         ]

    # # args_3 = [('gb_citers', 2),
    # #           ('g_min_size', 100),
    # #           ('gvar_start', 0),
    # #           ('g_bsnap_iter', 1),  # citers*4*bsnap_svrg
    # #           ('g_optim', ''),
    # #           ('g_optim_start', 5),
    # #           ('g_epoch', ''),
    # #           # ('g_stable', [1, 1e4]),  # [1000, 10]),
    # #           ]
    # # args += [OrderedDict(shared_args+gluster_args+args_3)]
    # args_4 = [('g_online', ''),
    #           ('g_osnap_iter', 10),  # [5, 10]),
    #           ('g_beta', .99),  # [.9, .99]),  # 1-lr (the desired lr)
    #           ('g_min_size', .001),  # 100x diff in probabilities
    #           # ('g_reinit', 'largest')
    #           ('g_init_mul', 2),
    #           ('g_reinit_iter', 10),
    #           ]
    # args += [OrderedDict(shared_args+gvar_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def imagenet_gvar_adam(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_half_adam' % dataset
    exclude = ['dataset', 'arch', 'epochs', 'weight_decay',
               'g_epoch', 'resume', 'ckpt_name', 'lr_decay_epoch',
               'gvar_log_iter', 'gvar_start', 'g_debug', 'g_bsnap_iter',
               'half_trained']
    shared_args = [('dataset', dataset),
                   # ('optim', 'sgd'),  # 'sgd', 'adam'
                   ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   # ('arch', 'resnet50'),
                   # ('batch_size', 128),
                   # ('test_batch_size', 64),
                   ('weight_decay', 1e-4),
                   #  ### pretrained
                   ('half_trained', ['']),
                   ('epochs', [45]),
                   # ('lr', [.01]),
                   ('lr_decay_epoch', ['15,45']),
                   # ('exp_lr', [None]),
                   ]
    shared_args += [('gvar_estim_iter', 10),  # 100
                    ('gvar_log_iter', 1000),
                    ('gvar_start', 0),
                    ('g_bsnap_iter', 5),
                    ('g_epoch', ''),
                    ]
    args_sgd = [('g_estim', ['sgd']),
                ('batch_size', [128, 64]),
                ('optim', [
                    ('sgd', OrderedDict([('lr', .01)])),
                    ('adam', OrderedDict([('lr', 1e-4)])),
                    # ('adamw', OrderedDict([('lr', 1e-4)])),
                ]),
                ]
    args += [OrderedDict(shared_args+args_sgd)]

    args_svrg = [('g_estim', ['svrg']),
                 ('batch_size', 64),
                 ('lr', 1e-4),  # 1e-4
                 ('g_optim', ''),
                 ('g_optim_start', 0),  # 5, 10, 20]),
                 ('g_mlr', [1, 2]),  # 0.1, 10]),
                 ('optim', ['adam']),
                 ]
    args += [OrderedDict(shared_args+args_svrg)]

    gluster_args = [('g_estim', 'gluster'),
                    ('g_nclusters', 64),  # [10, 100]),
                    ('g_debug', ''),
                    ('batch_size', 64),
                    ('lr', 1e-4),  # 1e-4
                    ('g_optim', ''),
                    ('g_optim_start', 1),  # 5, 10, 20]),
                    ('g_mlr', [1, 2]),  # 0.1, 10]),
                    ('optim', ['adam']),
                    ]

    # args_3 = [('gb_citers', 10)]
    # args += [OrderedDict(shared_args+snap_args+gluster_args+args_3)]
    args_4 = [('g_online', ''),
              ('g_osnap_iter', 10),
              ('g_beta', .99),  # 1-lr (the desired learning rate)
              ('g_min_size', .001),  # roughly .1*batch_size/nclusters
              # ('g_reinit', 'largest')  # default
              ('g_init_mul', 2),
              ('g_reinit_iter', 10),
              ]
    args += [OrderedDict(shared_args+gluster_args+args_4)]
    return args, log_dir, module_name, exclude


def main():
    args = []
    # args, log_dir, module_name, exclude = mnist_gvar(args)
    # args, log_dir, module_name, exclude = cifar10_gvar_cnn(args)
    # args, log_dir, module_name, exclude = mnist_gvar_dup(args)
    # args, log_dir, module_name, exclude = imagenet_pretrained_gvar(args)
    # args, log_dir, module_name, exclude = mnist_gvar_bs(args)
    # args, log_dir, module_name, exclude = cifar10_gvar_cnn_bs(args)
    # args, log_dir, module_name, exclude = svhn_gvar(args)
    # args, log_dir, module_name, exclude = imagenet_gvar(args)
    # args, log_dir, module_name, exclude = mnist_gvar_optim(args)
    # args, log_dir, module_name, exclude = cifar10_gvar_cnn_optim(args)
    # args, log_dir, module_name, exclude = cifar10_gvar_resnet_bs_optim(args)
    # args, log_dir, module_name, exclude = imagenet_gvar_half(args)
    # args, log_dir, module_name, exclude = cifar10_blup_inactive(args)
    # args, log_dir, module_name, exclude = cifar10_gvar_resnet_adam(args)
    # args, log_dir, module_name, exclude = cifar10_gvar_resnet_adam_svrg(args)
    args, log_dir, module_name, exclude = cifar10_gvar_adam_svrg_epoch0(args)
