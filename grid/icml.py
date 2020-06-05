from collections import OrderedDict


def logreg_2dvis(args):
    dataset = 'logreg'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_2dvis' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'weight_decay',
               'g_epoch', 'g_bsnap_iter', 'g_min_size']
    shared_args = [('dataset', dataset),
                   ('lr', [.0005]),
                   ('epochs', 2),
                   ('lr_decay_epoch', '1,2'),
                   ('lr_decay_rate', 0),
                   ('c_const', 5),
                   ('d_const', 1.5),
                   ('batch_size', 10),
                   ]
    gluster_args = [
        ('g_estim', 'gluster'),
        ('g_nclusters', [4]),
        ('g_debug', ''),
        ('gb_citers', 10),
        ('g_min_size', 1),
        ('gvar_start', 1),
        ('g_bsnap_iter', 10),
        ('g_epoch', ''),
        ('epoch_iters', [100, 200, 400]),
    ]

    args += [OrderedDict(shared_args+gluster_args)]

    return args, log_dir, module_name, exclude


def linreg(args):
    dataset = 'linreg'
    module_name = 'main.gvar'
    # log_dir = 'runs_%s_dim1000_niters1e5_bs10_lr5e-5_snr_10' % dataset
    # log_dir = 'runs_%s_online_dup10,100' % dataset
    # log_dir = 'runs_%s_online_dup5,100' % dataset
    # log_dir = 'runs_%s_online' % dataset
    log_dir = 'runs_%s_gvar_online_S1' % dataset
    exclude = ['dataset', 'lr', 'weight_decay', 'epochs', 'lr_decay_epoch',
               'optim', 'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'dim', 'niters']
    shared_args = [('dataset', dataset),
                   ('lr', 5e-5),
                   ('weight_decay', 0),
                   # ('epochs', 50),
                   # ('lr_decay_epoch', 50),
                   ('niters', 100000),
                   ('lr_decay_epoch', 100000),
                   ('dim', 1000),
                   ('num_train_data', 500),
                   # ('seed', [123, 456, 789]),
                   ('r2', 1),
                   ('snr', 2),  # 10
                   ('batch_size', 10),
                   # ### duplicate
                   # ('duplicate', '10,100'),
                   # ('duplicate', '100,100'),
                   # ('duplicate', '5,100'),
                   # ('duplicate', '5,1000'),
                   # ### online
                   ('linreg_online', '')
                   ]
    gvar_args = [
        # ('gvar_estim_iter', 10),  # default
        # ('gvar_log_iter', 100),  # default
        ('optim', 'sgd'),
        # ('g_optim', ''),
        # ('g_epoch', ''),
        ('gvar_start', 10),
        # ('g_optim_start', 1000),
        ('g_bsnap_iter', 1000),
        # ('g_mlr', [1, 2]),
    ]

    args_sgd = [
        ('g_estim', ['sgd']),
        # ('batch_size', [10, 20]),
        ('g_batch_size', [10, 20]),
    ]
    args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    shared_args2 = [
        # ('batch_size', 10),
    ]
    args_svrg = [
        ('g_estim', ['svrg']),
    ]
    args += [OrderedDict(shared_args+args_svrg+gvar_args+shared_args2)]

    gluster_args = [
        ('g_estim', 'gluster'),
        ('g_nclusters', [10]),
        ('g_debug', ''),
        ('gb_citers', 10),
        ('g_min_size', 1),
    ]

    args += [OrderedDict(shared_args+gluster_args+gvar_args+shared_args2)]

    return args, log_dir, module_name, exclude


def mnist_gvar(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar' % dataset
    # log_dir = 'runs_%s_gvar_imbalance' % dataset
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

    args_svrg = [
        ('g_estim', ['svrg']),
    ]
    args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    gluster_args = [
        ('g_estim', 'gluster'),
        ('g_nclusters', 128),  # [8, 128]),
        ('g_debug', ''),
        ('gb_citers', 10),
        ('g_min_size', 1),
    ]

    args += [OrderedDict(shared_args+gluster_args+gvar_args)]

    return args, log_dir, module_name, exclude


def cifar10_gvar(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    # log_dir = 'runs_%s_gvar_resnet32' % dataset
    # log_dir = 'runs_%s_gvar_resnet32_data_prod' % dataset
    # log_dir = 'runs_%s_gvar_resnet32_imbalance' % dataset
    # log_dir = 'runs_%s_gvar_resnet110' % dataset
    # log_dir = 'runs_%s_gvar_resnet32_sampling' % dataset
    # log_dir = 'runs_%s_gvar_resnet32_robust' % dataset
    # log_dir = 'runs_%s_gvar_resnet8_data' % dataset
    log_dir = 'runs_%s_gvar_resnet8_data_prod' % dataset
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
                   ('label_smoothing', [None, 0.1]),
                   ('data_aug', [None, '']),
                   ('corrupt_perc', [None, 20]),
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

    # args_sgd = [
    #     ('g_estim', ['sgd']),
    #     ('g_batch_size', [128, 256]),
    # ]
    # args += [tuple((OrderedDict(shared_args+args_sgd+gvar_args), disj_args))]

    args_svrg = [
        ('g_estim', ['svrg']),
    ]
    args += [tuple((OrderedDict(shared_args+args_svrg+gvar_args), disj_args))]

    # gluster_args = [
    #     ('g_estim', 'gluster'),
    #     ('g_nclusters', 128),  # [8, 128]),
    #     ('g_debug', ''),
    #     ('gb_citers', 3),  # 10),  # [2, 10, 20, 50]),
    #     ('g_min_size', 1),
    #     # ('wnoise', ''),
    #     # ('wnoise_stddev', [1e-2, 1e-3, 1e-4]),
    #     # ('g_avg', [50, 100]),  # [200, 500, 1000]),  # [10, 100]),
    #     # ('g_msnap_iter', 1),  # [1, 10]),
    #     # ('g_clip', 2),
    #     # ## sampling
    #     # ('g_imbalance', [None, '']),
    #     # ## robust
    #     # ('g_robust', ''),
    #     # ('g_robust_high', [0.05, 0.01]),  # [.9, .5, .1]),
    #     # ('g_robust_low', 1),  # [1, 1.1, 2]),
    # ]

    # args += [
    #     tuple((OrderedDict(shared_args+gluster_args+gvar_args), disj_args))]

    return args, log_dir, module_name, exclude


def cifar100_gvar(args):
    dataset = 'cifar100'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_resnet32_corrupt_dup' % dataset
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
                   ('duplicate', '10,10000'),
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

    # args_sgd = [
    #     ('g_estim', ['sgd']),
    #     ('g_batch_size', [128, 256]),
    # ]
    # args += [tuple((OrderedDict(shared_args+args_sgd+gvar_args), disj_args))]

    args_svrg = [
        ('g_estim', ['svrg']),
    ]
    args += [tuple((OrderedDict(shared_args+args_svrg+gvar_args), disj_args))]

    # gluster_args = [
    #     ('g_estim', 'gluster'),
    #     ('g_nclusters', 128),  # [8, 128]),
    #     ('g_debug', ''),
    #     ('gb_citers', 10),  # [2, 10, 20, 50]),
    #     ('g_min_size', 1),
    #     # ('wnoise', ''),
    #     # ('wnoise_stddev', [1e-2, 1e-3, 1e-4]),
    #     # ('g_avg', [50, 100]),  # [200, 500, 1000]),  # [10, 100]),
    #     # ('g_msnap_iter', 1),  # [1, 10]),
    #     # ('g_clip', 2),
    # ]

    # args += [
    #     tuple((OrderedDict(shared_args+gluster_args+gvar_args), disj_args))]

    return args, log_dir, module_name, exclude


def imagenet_gvar(args):
    dataset = 'imagenet'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_kahan' % dataset
    # log_dir = 'runs_%s_gvar_dup' % dataset
    # log_dir = 'runs_%s_gvar_dup10,100000' % dataset
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

    # args_sgd = [
    #     ('g_estim', ['sgd']),
    #     ('g_batch_size', [128, 256]),  # [64, 256]),  # [128, 256]),
    # ]
    # args += [tuple((OrderedDict(shared_args+args_sgd+gvar_args), disj_args))]

    args_svrg = [
        ('g_estim', ['svrg']),
    ]
    args += [tuple((OrderedDict(shared_args+args_svrg+gvar_args), disj_args))]

    # gluster_args = [
    #     ('g_estim', 'gluster'),
    #     # ('g_batch_size', [64, 256]),  # [128, 256]),
    #     ('g_nclusters', 128),  # 256),  # 128),  # [8, 128]),
    #     ('g_debug', ''),
    #     # # ### batch gluster
    #     ('gb_citers', 3),  # [2, 10, 20, 50]),
    #     ('g_min_size', 1),
    #     # ### kahan
    #     ('g_kahan', ''),
    #     # # ###
    #     # ('wnoise', ''),
    #     # ('wnoise_stddev', [1e-2, 1e-3, 1e-4]),
    #     # ('g_avg', [50, 100]),  # [200, 500, 1000]),  # [10, 100]),
    #     # ('g_msnap_iter', 1),  # [1, 10]),
    #     # ('g_clip', 2),
    #     # #### online gluster
    #     # ('g_online', ''),
    #     # ('g_osnap_iter', 10),
    #     # ('g_beta', .99),  # 1-lr (the desired learning rate)
    #     # ('g_min_size', .001),  # roughly .1*batch_size/nclusters
    #     # ('g_init_mul', 2),
    #     # ('g_reinit_iter', 10),
    # ]

    # args += [
    #     tuple((OrderedDict(shared_args+gluster_args+gvar_args), disj_args))]

    return args, log_dir, module_name, exclude


def nclusters(args):
    module_name = 'main.gvar'
    log_dir = 'runs_nclusters_randi_normal'
    exclude = ['lr', 'weight_decay', 'epochs', 'lr_decay_epoch',
               'optim', 'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'dim', 'niters', 'gvar_log_iter',
               'gvar_estim_iter']
    gvar_args = [
        ('gvar_estim_iter', 10),  # default 10
        ('gvar_log_iter', 1000),  # default 100
        ('optim', 'sgd'),
        ('gvar_start', 1000),  # 24100),  # 100),
        ('g_bsnap_iter', 10000),
    ]
    gluster_args = [
        ('g_estim', 'gluster'),
        # ('g_batch_size', [64, 256]),  # [128, 256]),
        ('g_nclusters', [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ('g_debug', ''),
        ('gb_citers', 1),  # [2, 10, 20, 50]), 5
        ('g_min_size', 1),
        ('g_rand_input', ''),
        # ('wnoise', ''),
        # ('wnoise_stddev', [1e-2, 1e-3, 1e-4]),
        # ('g_avg', [50, 100]),  # [200, 500, 1000]),  # [10, 100]),
        # ('g_msnap_iter', 1),  # [1, 10]),
        # ('g_clip', 2),
    ]

    shared_args = [('dataset', 'mnist'),
                   ('lr', .02),
                   ('arch', 'cnn'),  # ['mlp', 'cnn']),
                   # ('weight_decay', 0),
                   ('niters', 2000),
                   ('lr_decay_epoch', 50000),
                   # ('seed', [123, 456, 789]),
                   # ('nodropout', ''),
                   ('batch_size', 128),
                   ]
    args += [OrderedDict(shared_args+gluster_args+gvar_args)]

    # shared_args = [('dataset', 'cifar10'),
    #                ('lr', 0.1),  # .01),  # 0.1
    #                ('arch', 'resnet32'),
    #                # ('weight_decay', 0),
    #                ('niters', 2000),
    #                ('lr_decay_epoch', '40000,60000'),
    #                ('batch_size', 128),
    #                # ('seed', [123, 456, 789]),
    #                # ('label_smoothing', [None, 0.1]),
    #                # ('data_aug', [None, '']),
    #                # ('corrupt_perc', [None, 20]),
    #                ('duplicate', [None, '10,10000']),
    #                ]
    # args += [OrderedDict(shared_args+gluster_args+gvar_args)]

    # shared_args = [('dataset', 'imagenet'),
    #                ('lr', 0.1),  # .01),  # 0.1
    #                ('arch', 'resnet18'),
    #                ('weight_decay', 1e-4),
    #                ('niters', 2000),
    #                ('lr_decay_epoch', '40000,60000'),
    #                ('batch_size', 128),
    #                ('g_kahan', [None, '']),
    #                # ('seed', [123, 456, 789]),
    #                # ('label_smoothing', [None, 0.1]),
    #                # ('corrupt_perc', [None, 20]),
    #                ]
    # args += [OrderedDict(shared_args+gluster_args+gvar_args)]

    return args, log_dir, module_name, exclude


def cifar10_gvar_incluster(args):
    dataset = 'cifar10'
    module_name = 'main.gvar'
    log_dir = 'runs_%s_gvar_incluster' % dataset
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
                   ]),
                   # ('weight_decay', 0),
                   ('niters', 80000),
                   ('lr_decay_epoch', '40000,60000'),
                   # ('seed', [123, 456, 789]),
                   # ('label_smoothing', [None, 0.1]),
                   # ('data_aug', [None, '']),
                   # ('corrupt_perc', [None, 20]),
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

    # args_svrg = [
    #     ('g_estim', ['svrg']),
    # ]
    # args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    gluster_args = [
        ('g_estim', 'gluster'),
        ('g_nclusters', 128),  # [8, 128]),
        ('g_debug', ''),
        ('gb_citers', 3),  # [2, 10, 20, 50]),
        ('g_min_size', 1),
        ('g_incluster', [None, 0, 1, 10, 50, 100, 126, 127]),
        # ('wnoise', ''),
        # ('wnoise_stddev', [1e-2, 1e-3, 1e-4]),
        # ('g_avg', [50, 100]),  # [200, 500, 1000]),  # [10, 100]),
        # ('g_msnap_iter', 1),  # [1, 10]),
        # ('g_clip', 2),
    ]

    args += [
        tuple((OrderedDict(shared_args+gluster_args+gvar_args), disj_args))]

    return args, log_dir, module_name, exclude
