from collections import OrderedDict


def text8(args):
    dataset = 'text8'
    module_name = 'train'
    log_dir = 'runs_%s' % dataset
    exclude = [
        'cuda', 'data', 'dataset', 'n_layer', 'd_model', 'n_head', 'd_head',
        'd_inner', 'dropout', 'dropatt', 'warmup_step', 'max_step', 'tgt_len',
        'mem_len', 'attn_type', 'eval_tgt_len', 'log-interval',
        'batch_size', 'gvar_estim_iter', 'gvar_log_iter',
        'gvar_start', 'g_optim_start', 'g_epoch']
    shared_args = [
        ('cuda', ''),
        ('data', '../data/%s/' % dataset),
        ('dataset', dataset),
        ('n_layer', 6),
        ('d_model', 256),
        ('n_head', 8),
        ('d_head', 64),
        ('d_inner', 1024),
        ('dropout', 0.1),
        ('dropatt', 0.0),
        ('warmup_step', 0),
        ('max_step', 400000),
        ('tgt_len', 512),
        ('mem_len', 0),
        ('attn_type', 2),
        ('eval_tgt_len', 128),
        ('log-interval', 1),  # 200
        ('batch_size', 22),  # 44
        ('momentum', 0.9),
        ('weight_decay', 0),
        # ('gvar_log_iter', 200),
        # ('niters', 80000),
        # ('lr_decay_epoch', 40000, 60000),
    ]

    gvar_args = [
        # ('gvar_estim_iter', 10),  # default
        # ('gvar_log_iter', 100),  # default
        ('gvar_start', 0),
        ('g_optim', ''),
        ('g_optim_start', 0),  # [0, 10, 20]),
        ('g_epoch', ''),
    ]
    args_sgd = [('g_estim', ['sgd']),
                ('optim', 'adam'),
                ('lr', 0.0001),
                ]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    # args_kfac = [('g_estim', ['sgd']),
    #              ('optim', ['kfac']),
    #              ('lr', 0.02),  # [.1, .05, .02, .01]),
    #              ('kf_damping',  0.03),  # [0.03, 0.02, 0.01, 0.005, 0.002]),
    #              ]
    # args += [OrderedDict(shared_args+gvar_args+args_kfac)]

    args_ntk = [('g_estim', ['ntk']),
                ('optim', 'adam'),
                ('lr', 0.0001),
                ('ntk_damping', [5e-2, 3e-2, 1e-2]),  # , 5e-3, 2e-3, 1e-3]),
                # ('ntk_cpu', ''),
                ]
    args += [OrderedDict(shared_args+gvar_args+args_ntk)]
    return args, log_dir, module_name, exclude
