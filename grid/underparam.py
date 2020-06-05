from collections import OrderedDict
import numpy as np


def rf(args):
    dataset = 'rf'
    module_name = 'main.gvar'
    # log_dir = 'runs_%s_gvar_hs1000_hsN' % dataset
    # log_dir = 'runs_%s_gvar_hs1000_hsN_dup5,0.1' % dataset
    # log_dir = 'runs_rf/runs_%s_gvar_hs1000_hsN_dup5,0.5' % dataset
    log_dir = 'runs_rf/runs_%s_gvar_hs1000_hsN_dup5,0.9' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch',
               'g_optim', 'g_epoch', 'gvar_start', 'g_optim_start',
               'g_bsnap_iter', 'niters', 'momentum']
    tau = np.concatenate((np.array([0.1]),
                          (np.arange(0, 3, 0.5)+0.5),
                          np.array([5, 10])))
    shared_args = [('dataset', dataset),
                   # ('lr', [1e-1, 1e-2, 1e-3]),
                   ('lr', [1e-2]),
                   ('momentum', 0),
                   ('weight_decay', 1e-4),
                   # ('epochs', 50),
                   # ('lr_decay_epoch', 50),
                   ('niters', 5000),
                   ('lr_decay_epoch', 100000),
                   ('batch_size', 10),
                   ('seed', [123, 456, 789]),
                   ('dim', [100, 10000]),
                   # ('num_train_data', [100, 500, 1000, 1500, 2000]),
                   # ('num_train_data', list(np.arange(100, 2000, 500))),
                   # ('teacher_hidden', list(np.arange(100, 2000, 500))),
                   ('student_hidden', [1000]),
                   ('num_train_data', [int(1000/t) for t in tau]),
                   ('teacher_hidden', [100, 10000]),
                   # ('teacher_hidden', [int(1000/t) for t in tau]),
                   # ('duplicate', ['5,0.1'])
                   # ('duplicate', ['5,0.5'])
                   ('duplicate', ['5,0.9'])
                   ]
    # print(len(tau))
    # print(shared_args)
    gvar_args = [
        # ('gvar_estim_iter', 10),  # default
        # ('gvar_log_iter', 100),  # default
        ('optim', 'sgd'),
        ('gvar_start', 10),
        ('g_bsnap_iter', 1000),
        # ('g_optim', ''),
        # ('g_epoch', ''),
        # ('g_optim_start', 1000),
    ]

    args_sgd = [
        ('g_estim', ['sgd']),
        ('g_batch_size', [10, 20]),
    ]
    args += [OrderedDict(shared_args+args_sgd+gvar_args)]

    args_svrg = [
        ('g_estim', ['svrg']),
    ]
    args += [OrderedDict(shared_args+args_svrg+gvar_args)]

    gluster_args = [
        ('g_estim', 'gluster'),
        ('g_nclusters', [10]),
        ('g_debug', ''),
        ('gb_citers', 3),
        ('g_min_size', 1),
    ]

    args += [OrderedDict(shared_args+gluster_args+gvar_args)]

    return args, log_dir, module_name, exclude
