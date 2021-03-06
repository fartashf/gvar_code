from collections import OrderedDict


def mnist_dmom(args):
    args = [OrderedDict([('optim', ['sgd']),
                         ('lr', [0.1, 0.01, 0.001]),
                         ('momentum', [0.1, 0.5, 0.9, 0.99]),
                         ]),
            OrderedDict([('optim', ['dmom']),
                         ('lr', [0.1, 0.01, 0.001]),
                         ('dmom', [0.1, 0.5, 0.9, 0.99]),
                         ('momentum', [0.1, 0.5, 0.9, 0.99]),
                         ])]
    return args


def mnist_sgd_hparam(args):
    # sgd
    args += [OrderedDict([('optim', ['dmom']),
                          ('lr', [0.1, 0.01, 0.001]),
                          ('dmom', [0]),
                          ('momentum', [0, 0.5, 0.9, 0.99]),
                          ])]
    return args


def mnist_low_theta(args):
    args += [OrderedDict([('optim', ['dmom']),
                          ('lr', [0.1, 0.01, 0.001]),
                          ('dmom', [0.5, 0.9, 0.99]),
                          ('momentum', [0., 0.5, 0.9, 0.95, 0.99]),
                          # ('dmom_interval', [1, 3, 5]),
                          # ('dmom_temp', [0, 1, 2]),
                          # ('alpha', ['jacobian']),
                          # ('jacobian_lambda', [
                          # ('dmom_theta', [0, 10]),
                          ('low_theta', [1e-4, 1e-2]),
                          ])]
    return args


def logreg1(args):
    args += [OrderedDict([('dataset', ['logreg']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ])]
    return args


def mnist_sampler_alpha(args):
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['ggbar_abs', 'ggbar_max0']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [.01]),
                          ('sampler_alpha_th', [.00000001, .000001, .00001]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['normg']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [1]),
                          ('sampler_alpha_th', [.00000001, .000001, .00001]),
                          ])]
    return args


def mnist_weighted(args):
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.001]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('batch_size', [12, 13]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          # ('dmom_interval', [1, 3, 5]),
                          # ('dmom_temp', [0, 1, 2]),
                          ('alpha', ['normg',
                                     'ggbar_abs', 'ggbar_max0', 'loss']),
                          # ('jacobian_lambda', [
                          # ('dmom_theta', [0, 10]),
                          # ('low_theta', [1e-4, 1e-2]),
                          ('alpha_norm', ['none', 'sum']),
                          # 'sum_batch', 'exp_batch']),
                          # ('weight_decay', [0, 5e-4]),
                          # ('wmomentum', [False, True]),
                          # ('arch', ['mlp', 'convnet']),
                          # ('norm_temp', [1, 2, 10, 100]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['normg', 'ggbar_abs', 'ggbar_max0',
                                     'loss']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [1, .01]),
                          ])]
    return args


def mnist_alpha_perc(args):
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['normg']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [1]),
                          ('sampler_alpha_perc', [30, 50, 70, 90]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['ggbar_abs', 'ggbar_max0']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [.01]),
                          ('sampler_alpha_perc', [30, 50, 70, 90]),
                          ])]
    return args


def mnist_sampler_hparam_search(args):
    # sampler hyper param search
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['normg']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [1]),
                          ('sampler', ['']),
                          ('sampler_w2c', ['1_over', 'log']),
                          ('sampler_max_count', [10, 20, 50]),
                          ('sampler_start_epoch', [1, 3, 10]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['ggbar_abs']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [.01]),
                          ('sampler', ['']),
                          ('sampler_w2c', ['1_over', 'log']),
                          ('sampler_max_count', [10, 20, 50]),
                          ('sampler_start_epoch', [1, 3, 10]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['normg']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [1]),
                          ('sampler', ['']),
                          ('sampler_w2c', ['1_over']),
                          ('sampler_max_count', [50]),
                          ('sampler_start_epoch', [1]),
                          ('sampler_lr_update', ['']),
                          ('sampler_lr_window', [20, 40, 100]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['ggbar_abs', 'normg']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [.01]),
                          ('sampler', ['']),
                          ('sampler_w2c', ['1_over']),
                          ('sampler_max_count', [1000, 10000]),
                          ('sampler_start_epoch', [1]),
                          ('sampler_lr_update', ['']),
                          ('sampler_lr_window', [20, 40]),
                          ])]
    return args


def logreg_figure(args):
    # logreg figure
    args += [OrderedDict([('dataset', ['logreg']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['normg', 'ggbar_abs']),
                          # ('alpha_norm', ['exp']),
                          # ('norm_temp', [.0001]),
                          ('alpha_norm', ['sum', 'sum_class']),
                          ])]
    return args


def div_len(args):
    # ##### /len
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['exp', 'none']),
                          ('norm_temp', [1]),
                          ('divlen', [0, 1]),
                          ('sampler_alpha_perc', [0, 90]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['normg']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [1]),
                          ('divlen', [0, 1]),
                          ('sampler_alpha_perc', [0, 90]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['ggbar_abs']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [.01]),
                          ('divlen', [0, 1]),
                          ('sampler_alpha_perc', [0, 90]),
                          ])]
    return args


def mnist_weighted_bugfix(args):
    # wmomentum fixed
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['exp']),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0., 0.5, 0.9]),
                          ('momentum', [0.9]),
                          ('wmomentum', [0., 0.5, 0.9]),
                          ('alpha', ['normg', 'ggbar_abs']),
                          ('alpha_norm', ['exp']),
                          ('norm_temp', [100, 10, 1, .1]),
                          ])]
    return args


def mnist_sgd_equivalent(args):
    # sgd equivalent
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['sgd']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ])]
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('wmomentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['exp', 'sum', 'none']),
                          ])]
    return args


def mnist_sampler(args):
    # sampler
    dataset = 'mnist'
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['sgd']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('wmomentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['none']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['none']),
                          ('sampler', [''])
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          # ('alpha_norm', ['none']),
                          ('sampler', ['']),
                          ('alpha', ['ggbar_abs', 'normg', 'loss']),  #
                          ('dmom', [0., 0.5, 0.9]),
                          # ('wmomentum', [0., 0.5, 0.9]),
                          # ('norm_temp', [10, 1., .1]),
                          ('sampler_w2c', ['linear']),
                          # ('sampler_max_count', [20, 50]),
                          # ('sampler_start_epoch', [1, 2]),
                          ('sampler_params',
                           ['50,90,20,100', '50,99,20,100', '50,90,40,100']),
                          ])]
    return args


def cifar10_sampler(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_sampler' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['sgd']),
                          ('lr', [0.01]),
                          ('momentum', [0.9]),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('wmomentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['none']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['none']),
                          ('sampler', [''])
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          # ('alpha_norm', ['none']),
                          ('sampler', ['']),
                          ('sampler_params',
                           ['20,70,20,100', '20,70,40,100']),
                          ('alpha', ['ggbar_abs', 'normg', 'loss']),  #
                          ('dmom', [0., 0.5, 0.9]),
                          # ('wmomentum', [0., 0.5, 0.9]),
                          # ('norm_temp', [10, 1., .1]),
                          ('sampler_w2c', ['linear']),
                          # ('sampler_max_count', [20, 50]),
                          # ('sampler_start_epoch', [1, 2]),
                          ])]
    return args, log_dir


def cifar10_jvp(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_jvp' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['sgd']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('wmomentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['none']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['none']),
                          ('sampler', [''])
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('lr', [0.01]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['one']),
                          ('sampler_w2c', ['linear']),
                          ('sampler_params', ['20,70,20,100']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('lr', [0.01]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', [  # 'loss',
                                     'ggbar_abs', 'loss', 'gFigbar_abs',
                                     'lgg', 'gtheta_abs',
                                     ]),
                          # ('alpha', ['gFgbar_abs', 'gFigbar_abs']),
                          ('dmom', [0., 0.5, 0.9]),
                          ('sampler_w2c', ['linear']),
                          ('sampler_params',
                           ['20,70,20,100']),
                          ])]
    return args, log_dir


def cifar10_resnet(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_resnet_explr' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['sgd']),
                          ('epochs', [200]),
                          ('lr', [0.1]),
                          ('weight_decay', [1e-4]),
                          ('lr_decay_epoch', ['100,150']),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('epochs', [200]),
                          ('lr', [0.1]),
                          ('weight_decay', [1e-4]),
                          ('lr_decay_epoch', ['100,150']),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['one']),
                          # ('sampler_w2c', ['linear']),
                          ('sampler_params', ['20,70,50,200']),
                          # ('sampler_repetition', ['']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('epochs', [200]),
                          ('lr', [0.1]),
                          ('weight_decay', [1e-4]),
                          ('lr_decay_epoch', ['100,150']),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['loss']),
                          ('dmom', [0., 0.5, 0.9]),
                          # ('sampler_w2c', ['linear']),
                          ('sampler_params', ['20,70,50,200']),
                          # ('sampler_repetition', ['']),
                          ])]
    return args, log_dir


def mnist_jvp(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_jvp' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['sgd']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('lr', [0.01]),
                          ('seed', [1, 2]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('sampler_params', ['50,90,20,100']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('lr', [0.01]),
                          ('seed', [1, 2]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['ggbar_abs', 'loss']),
                          ('dmom', [0., 0.5, 0.9]),
                          ('sampler_params', ['50,90,20,100']),
                          ])]
    return args, log_dir


def cifar10_resnet_explr(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_resnet_explr' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['sgd']),
                          ('epochs', [200]),
                          ('lr', [0.1]),
                          ('weight_decay', [1e-4]),
                          # ('lr_decay_epoch', ['100,150']),
                          ('lr_decay_epoch', [100]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('exp_lr', ['']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('epochs', [200]),
                          ('lr', [0.1]),
                          ('weight_decay', [1e-4]),
                          # ('lr_decay_epoch', ['100,150']),
                          ('lr_decay_epoch', [100]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['one']),
                          # ('sampler_w2c', ['linear']),
                          ('sampler_params', ['20,70,50,200']),
                          # ('sampler_repetition', ['']),
                          ('exp_lr', ['']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('epochs', [200]),
                          ('lr', [0.1]),
                          ('weight_decay', [1e-4]),
                          # ('lr_decay_epoch', ['100,150']),
                          ('lr_decay_epoch', [100]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          # ('sampler_w2c', ['linear']),
                          ('sampler_params', ['20,70,50,200']),
                          # ('sampler_repetition', ['']),
                          ('exp_lr', ['']),
                          ])]
    return args, log_dir


def mnist_explr(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_explr' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['sgd']),
                          ('lr', [0.01]),
                          ('lr_decay_epoch', [50]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('exp_lr', ['']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('lr', [0.01]),
                          ('lr_decay_epoch', [50]),
                          ('seed', [1, 2]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('sampler_params', ['50,90,20,100']),
                          ('exp_lr', ['']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('optim', ['dmom_jvp']),
                          ('lr', [0.01]),
                          ('lr_decay_epoch', [50]),
                          ('seed', [1, 2]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['ggbar_abs', 'loss']),
                          ('dmom', [0., 0.5, 0.9]),
                          ('sampler_params', ['50,90,20,100']),
                          ('exp_lr', ['']),
                          ])]
    return args, log_dir


def svhn_resnet(args):
    # from wide-resnet https://arxiv.org/pdf/1605.07146.pdf
    dataset = 'svhn'
    log_dir = 'runs_%s' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['sgd']),
                          ('epochs', [160]),
                          ('lr', [0.01]),
                          ('weight_decay', [5e-4]),
                          ('lr_decay_epoch', [80]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('exp_lr', ['']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('epochs', [160]),
                          ('lr', [0.01]),
                          ('weight_decay', [5e-4]),
                          ('lr_decay_epoch', [80]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['one']),
                          ('sampler_params', ['50,90,50,200']),
                          ('exp_lr', ['']),
                          ])]
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('epochs', [160]),
                          ('lr', [0.01]),
                          ('weight_decay', [5e-4]),
                          ('lr_decay_epoch', [80]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ('sampler_params', ['50,90,50,200']),
                          ('exp_lr', ['']),
                          ])]
    return args, log_dir


def svhn_ggbar(args):
    # from wide-resnet https://arxiv.org/pdf/1605.07146.pdf
    dataset = 'svhn'
    log_dir = 'runs_%s' % dataset
    args += [OrderedDict([('dataset', [dataset]),
                          ('arch', ['resnet32']),
                          ('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('epochs', [160]),
                          ('lr', [0.01]),
                          ('weight_decay', [5e-4]),
                          ('lr_decay_epoch', [80]),
                          ('momentum', [0.9]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['ggbar_abs']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ('sampler_params', ['50,90,50,200']),
                          ('exp_lr', ['']),
                          ])]
    return args, log_dir


def imagenet(args):
    # from resnet https://arxiv.org/pdf/1512.03385.pdf
    dataset = 'imagenet'
    log_dir = 'runs_%s' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet18']),
                   ('epochs', [60]),
                   ('lr', [0.1]),
                   ('weight_decay', [1e-4]),
                   ('lr_decay_epoch', [30]),
                   ('momentum', [0.9]),
                   ('exp_lr', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd']),
                          ] + shared_args)]
    # args += [OrderedDict([('optim', ['dmom_jvp']),
    #                       ('seed', [1, 2]),
    #                       ('alpha_norm', ['sum']),
    #                       ('sampler', ['']),
    #                       ('alpha', ['one']),
    #                       ('sampler_params', ['50,90,50,200']),
    #                       ] + shared_args)]
    args += [OrderedDict([('optim', ['dmom_jvp']),
                          # ('seed', [1, 2]),
                          ('alpha_norm', ['sum']),
                          ('sampler', ['']),
                          ('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ('sampler_params', ['20,90,5,20']),
                          ] + shared_args)]
    return args, log_dir


def mnist_linear2(args):
    # the drop in the old results at the beginning was due to maxw=10
    dataset = 'mnist'
    log_dir = 'runs_%s_linear2_droplast' % dataset
    shared_args = [('dataset', [dataset]),
                   ('lr', [0.01]),
                   ('lr_decay_epoch', [50]),
                   ('momentum', [0.9]),
                   ('exp_lr', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd'])] + shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    ('seed', [1, 2]),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_params', ['50,90,2,200']),
                    ('sampler_w2c', ['linear2']),
                    ]
    args += [OrderedDict([('alpha', ['one']),
                          ]+shared_args)]
    args += [OrderedDict([('alpha', ['ggbar_abs', 'loss']),
                          ('dmom', [0., 0.5, 0.9]),
                          ]+shared_args)]
    return args, log_dir


def cifar10_linear2(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_linear2_droplast' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet32']),
                   ('momentum', [0.9]),
                   ('epochs', [200]),
                   ('lr', [0.1]),
                   ('weight_decay', [1e-4]),
                   ('lr_decay_epoch', [100]),
                   ('exp_lr', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd']),
                          ]+shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    ('seed', [1, 2]),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_w2c', ['linear2']),
                    ('sampler_params', ['20,90,2,200']),
                    ]
    args += [OrderedDict([('alpha', ['one']),
                          ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def svhn_linear2(args):
    # from wide-resnet https://arxiv.org/pdf/1605.07146.pdf
    dataset = 'svhn'
    log_dir = 'runs_%s_linear2_droplast' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet32']),
                   ('epochs', [160]),
                   ('lr', [0.01]),
                   ('weight_decay', [5e-4]),
                   ('lr_decay_epoch', [80]),
                   ('momentum', [0.9]),
                   ('exp_lr', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd'])]+shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    ('seed', [1, 2]),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_w2c', ['linear2']),
                    ('sampler_params', ['50,90,2,200']),
                    ]
    args += [OrderedDict([('alpha', ['one']),
                          ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_bs(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_bs_steplr_maxw' % dataset
    shared_args = [('dataset', [dataset]),
                   ('batch_size', [32, 128, 1024]),
                   # ('lr', [0.01]),
                   ('lr', [.001, .005, .01, .05, .1, .2, .5]),
                   # ('lr_decay_epoch', [50]),
                   ('lr_decay_epoch', [30]),
                   ('momentum', [0.9]),
                   # ('exp_lr', ['']),
                   ('log_nex', ['']),
                   # ('lr_scale', ['']),
                   ]
    # args += [OrderedDict([('optim', ['sgd'])] + shared_args)]
    args += [OrderedDict([('optim', ['dmom_jvp']),
                          ('alpha', ['one']),
                          ]+shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_params', ['50,90,2,200']),
                    ('sampler_w2c', ['linear2']),
                    ('sampler_maxw', [1]),
                    ]
    # args += [OrderedDict([('alpha', ['one']),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def cifar10_bs(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_bs_steplr_maxw' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet32']),
                   ('momentum', [0.9]),
                   ('epochs', [200]),
                   ('batch_size', [
                       # (32, OrderedDict([('lr', [.01, .05, .1])])),
                       # (512, OrderedDict([('lr', [.1, .2, .5])])),
                       (1024, OrderedDict([('lr', [.5])])),
                   ]),
                   # ('lr', [0.1]),
                   ('weight_decay', [1e-4]),
                   # ('lr_decay_epoch', [100]),
                   ('lr_decay_epoch', ['100,150']),
                   # ('exp_lr', ['']),
                   ('log_nex', ['']),
                   # ('lr_scale', ['']),
                   ]
    # args += [OrderedDict([('optim', ['sgd']),
    #                       ]+shared_args)]
    args += [OrderedDict([('optim', ['dmom_jvp']),
                          ('alpha', ['one']),
                          ]+shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_w2c', ['linear2']),
                    ('sampler_params', ['20,90,2,200']),
                    ('sampler_maxw', [1]),
                    # ('seed', [1, 2]),
                    ]
    # args += [OrderedDict([('alpha', ['one']),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def imagenet_linear2_bs(args):
    # from resnet https://arxiv.org/pdf/1512.03385.pdf
    dataset = 'imagenet'
    log_dir = 'runs_%s_droplast_bs' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet18']),
                   ('epochs', [60]),
                   # ('lr', [0.1]),
                   ('batch_size', [
                       (256, OrderedDict([('lr', [.1]),
                                          ('workers', [12])])),
                       (1024, OrderedDict([('lr', [.5]),
                                           ('workers', [12])])),
                   ]),
                   # ('weight_decay', [1e-4]),  # default
                   ('lr_decay_epoch', [30]),
                   # ('momentum', [0.9]),  # default
                   ('exp_lr', ['']),
                   ('log_nex', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd']),
                          ] + shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    # ('seed', [1, 2]),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    # ('sampler_w2c', ['linear2']),  # now default
                    ('sampler_params', ['20,90,2,50']),
                    # ('sampler_maxw', [1]),  # now default
                    ]
    # args += [OrderedDict([('alpha', ['one']),
    #                       ] + shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ] + shared_args)]
    return args, log_dir


def svhn_bs(args):
    # from wide-resnet https://arxiv.org/pdf/1605.07146.pdf
    dataset = 'svhn'
    log_dir = 'runs_%s_bs' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet32']),
                   ('epochs', [160]),
                   # ('lr', [0.01]),
                   ('batch_size', [
                       (128, OrderedDict([('lr', [.01]),
                                          ('workers', [12])])),
                       (1024, OrderedDict([('lr', [.1]),
                                           ('workers', [12])])),
                   ]),
                   # ('weight_decay', [5e-4]),  # default
                   ('lr_decay_epoch', [80]),
                   # ('momentum', [0.9]),  # default
                   ('exp_lr', ['']),
                   ('log_nex', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd'])]+shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    # ('seed', [1, 2]),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    # ('sampler_w2c', ['linear2']),  # default
                    ('sampler_params', ['50,90,2,200']),
                    ]
    # args += [OrderedDict([('alpha', ['one']),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_autoexp(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_autoexp' % dataset
    shared_args = [('dataset', [dataset]),
                   ('lr', [.01]),
                   ('lr_decay_epoch', [50]),
                   ('momentum', [0.9]),
                   ('exp_lr', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd'])] + shared_args)]
    shared_args += [('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_params', ['50,99,2,200']),
                    ('sampler_w2c', ['autoexp']),
                    ('sampler_maxw', [1]),
                    ]
    args += [OrderedDict([('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('alpha', ['one']),
                          ]+shared_args)]
    args += [OrderedDict([('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('alpha', ['loss']),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def cifar10_autoexp(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_autoexp' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet32']),
                   ('momentum', [0.9]),
                   ('epochs', [200]),
                   ('lr', [0.1]),
                   ('weight_decay', [1e-4]),
                   ('lr_decay_epoch', [100]),
                   ('exp_lr', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd'])] + shared_args)]
    shared_args += [('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_params', ['50,99,2,200']),
                    ('sampler_w2c', ['autoexp']),
                    ('sampler_maxw', [1]),
                    ]
    args += [OrderedDict([('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('alpha', ['one']),
                          ]+shared_args)]
    args += [OrderedDict([('optim', ['dmom_jvp']),
                          ('seed', [1, 2]),
                          ('alpha', ['loss']),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_linrank(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_linrank_wnoise' % dataset
    shared_args = [('dataset', [dataset]),
                   ('lr', [0.01]),
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', [50])])),
                       # (40, OrderedDict([('lr_decay_epoch', [20])])),
                       (100, OrderedDict([('lr_decay_epoch', [100])])),
                   ]),
                   # ('lr_decay_epoch', [30]),
                   ('momentum', [0.9]),
                   # ('exp_lr', ['']),
                   ('wnoise', ['']),
                   # ('wnoise_stddev', [1., .1, .01, .001]),
                   ]
    args += [OrderedDict([('optim', ['sgd'])] + shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    # ('seed', [1, 2]),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_params', ['50,90,2,200']),
                    ('sampler_w2c', ['linrank']),
                    ]
    # args += [OrderedDict([('alpha', ['one']),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def cifar10_linrank(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_linrank_explr_wnoise' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet32']),
                   ('epochs', [
                       # (200, OrderedDict([('lr_decay_epoch', ['100,150'])])),
                       # (100, OrderedDict([('lr_decay_epoch', ['50,75'])])),
                       (200, OrderedDict([('exp_lr', ['']),
                                          ('lr_decay_epoch', [100])])),
                       # (100, OrderedDict([('exp_lr', ['']),
                       #                    ('lr_decay_epoch', [50])])),
                   ]),
                   ('batch_size', [
                       (32, OrderedDict([('lr', [.1])])),
                       (1024, OrderedDict([('lr', [.5])])),
                   ]),
                   # ('lr', [0.1]),
                   ('weight_decay', [1e-4]),
                   # ('lr_decay_epoch', [100]),
                   # ('lr_decay_epoch', ['100,150']),
                   ('log_nex', ['']),
                   # ('data_aug', ['']),
                   ('wnoise', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd']),
                          ]+shared_args)]
    # args += [OrderedDict([('optim', ['dmom_jvp']),
    #                       ('alpha', ['one']),
    #                       ]+shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_w2c', ['linrank']),
                    ('sampler_params', ['50,90,2,200']),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [2, 5]),
                    ]
    # args += [OrderedDict([('alpha', ['one']),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def svhn_linrank(args):
    # Same as cifar10 resnet32
    dataset = 'svhn'
    log_dir = 'runs_%s_linrank_explr_wnoise' % dataset
    shared_args = [('dataset', [dataset]),
                   ('arch', ['resnet32']),
                   ('epochs', [
                       # (160, OrderedDict([('lr_decay_epoch', ['80,120'])])),
                       # (120, OrderedDict([('lr_decay_epoch', ['60,90'])])),
                       (160, OrderedDict([('exp_lr', ['']),
                                          ('lr_decay_epoch', [80])])),
                   ]),
                   ('batch_size', [
                       (128, OrderedDict([('lr', [.1])])),
                       (1024, OrderedDict([('lr', [.5])])),
                   ]),
                   ('weight_decay', [1e-4]),
                   # ('lr', [0.1]),
                   ('log_nex', ['']),
                   ('wnoise', ['']),
                   ]
    args += [OrderedDict([('optim', ['sgd']),
                          ]+shared_args)]
    # args += [OrderedDict([('optim', ['dmom_jvp']),
    #                       ('alpha', ['one']),
    #                       ]+shared_args)]
    shared_args += [('optim', ['dmom_jvp']),
                    ('alpha_norm', ['sum']),
                    ('sampler', ['']),
                    ('sampler_w2c', ['linrank']),
                    ('sampler_params', ['50,90,2,200']),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [1, 5]),
                    ]
    # args += [OrderedDict([('alpha', ['one']),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', ['loss']),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_expsnooze(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_expsnooze_ad_hist_hparam_nonorm' % dataset
    shared_args = [('dataset', dataset),
                   ('lr', 0.01),
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', 50)])),
                       # (40, OrderedDict([('lr_decay_epoch', 20)])),
                       (100, OrderedDict([('lr_decay_epoch', 100)])),
                   ]),
                   # ('lr_decay_epoch', 30),
                   ('momentum', 0.9),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('wnoise_stddev', [1., .1, .01, .001]),
                   ]
    args += [OrderedDict([('optim', 'sgd')] + shared_args)]
    shared_args += [('optim', 'dmom_ns'),
                    # ('seed', [1, 2]),
                    ('alpha_norm', 'none'),
                    # ('alpha_norm', ['sum', 'none']),
                    ('sampler', ''),
                    # ('min_batches', 10),
                    ('sampler_w2c', [
                        ('linrank',
                         OrderedDict([('sampler_params',
                                       ['50,90,2,200']
                                       )])),
                        ('expsnz_tau',
                         OrderedDict([('sampler_params',
                                       # [1e-3, 1e-4]
                                       [1e-2, 1e-3]
                                       ),
                                      ])),
                        ('expsnz_tauXstd',
                         OrderedDict([('sampler_params',
                                       [.2, .1, .05]
                                       ),
                                      ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [2., 1., .01]
                                       [2., 1.]
                                       ),
                                      ])),
                        # ('exp_snooze_th',
                        #  OrderedDict([('sampler_params',
                        #                # [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
                        #                # [1e-3, 1e-4, 1e-5, 1e-6]
                        #                [1e-3, 1e-4]
                        #                # [1e-5, 1e-6]
                        #                ),
                        #               ])),
                        # ('exp_snooze_th',
                        #  OrderedDict([('sampler_params',
                        #                [.2, .1, .05]
                        #                ),
                        #               ('tauXstd', '')
                        #               ])),
                        # ('exp_snooze_th',
                        #  OrderedDict([('sampler_params',
                        #                [.05, .01, .001]
                        #                ),
                        #               ('tauXstdL', '')
                        #               ])),
                        # ('expsnz_cumsum',
                        #  OrderedDict([('sampler_params',
                        #                [.1, .01, .001, .0001]
                        #                ),
                        #               ])),
                        # ('expsnz_tauXmu',
                        #  OrderedDict([('sampler_params',
                        #                [2., 1., .5, .2, .1]
                        #                ),
                        #               ])),
                        # ('exp_snooze_med',
                        #  OrderedDict([('sampler_params',
                        #                # [1e-1, 1e-2, 1e-3]
                        #                [1., .5]
                        #                )])),
                        # ('exp_snooze_mean',
                        #  OrderedDict([('sampler_params',
                        #                # [1e-1, 1e-2, 1e-3]
                        #                [1., .5]
                        #                )])),
                        # ('exp_snooze_perc',
                        #  OrderedDict([('sampler_params',
                        #                ['10,1.', '10,10', '1,10']
                        #                )])),
                        # ('exp_snooze_lin',
                        #  OrderedDict([('sampler_params',
                        #                ['50,1', '50,2', '50,4',
                        #                 '10,1', '10,2', '10,4']
                        #                )])),
                        # ('exp_snooze_expdec',
                        #  OrderedDict([('sampler_params',
                        #                ['1e-3,100', '1e-3,50']
                        #                )])),
                        # ('exp_snooze_stepdec',
                        #  OrderedDict([('sampler_params',
                        #                ['1e-3,50']
                        #                )])),
                        # ('exp_snooze_expinc',
                        #  OrderedDict([('sampler_params',
                        #                ['1e-3,50']
                        #                )])),
                        # ('exp_snooze_stepinc',
                        #  OrderedDict([('sampler_params',
                        #                ['1e-3,50']
                        #                )])),
                    ]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', 'loss'),
                          ('dmom', [0., 0.9]),
                          # ('dmom', 0.9),
                          ]+shared_args)]
    return args, log_dir


def cifar10_expsnooze(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_expsnooze_ad_hist_hparam_nonorm' % dataset
    shared_args = [('dataset', dataset),
                   ('arch', 'resnet32'),
                   ('epochs', [
                       # (200, OrderedDict([('lr_decay_epoch', '100,150')])),
                       # (100, OrderedDict([('lr_decay_epoch', '50,75')])),
                       (200, OrderedDict([('exp_lr', ''),
                                          ('lr_decay_epoch', 100)])),
                       # (100, OrderedDict([('exp_lr', ''),
                       #                    ('lr_decay_epoch', 50)])),
                   ]),
                   # ('batch_size', [
                   #     (32, OrderedDict([('lr', .1)])),
                   #     (1024, OrderedDict([('lr', .5)])),
                   # ]),
                   # ('log_nex', ''),
                   ('lr', 0.1),
                   ('weight_decay', 1e-4),
                   # ('lr_decay_epoch', 100),
                   # ('lr_decay_epoch', '100,150'),
                   # ('data_aug', ''),
                   # ('wnoise', [None, '']),
                   ]
    args += [OrderedDict([('optim', 'sgd'),
                          ]+shared_args)]
    # args += [OrderedDict([('optim', 'dmom_jvp'),
    #                       ('alpha', 'one'),
    #                       ]+shared_args)]
    shared_args += [('optim', 'dmom_ns'),
                    ('alpha_norm', 'sum'),
                    ('sampler', ''),
                    ('sampler_w2c', [
                        ('linrank',
                         OrderedDict([('sampler_params',
                                       ['50,90,2,200']
                                       )])),
                        ('expsnz_tau',
                         OrderedDict([('sampler_params',
                                       [1e-3]
                                       )])),
                        ('expsnz_tauXstd',
                         OrderedDict([('sampler_params',
                                       # [.1, .01]
                                       [.1]
                                       ),
                                      ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [.1, .01]
                                       [.01]
                                       ),
                                      ])),
                        # ('exp_snooze_th',
                        #  OrderedDict([('sampler_params',
                        #                [5e-3, 1e-3, 2e-3]
                        #                # [1e-3, 1e-4]
                        #                # [1e-5, 1e-6]
                        #                )])),
                        # ('exp_snooze_th',
                        #  OrderedDict([('sampler_params',
                        #                [.1, .05, .2]
                        #                ),
                        #               ('tauXstd', '')])),
                        # ('exp_snooze_expdec',
                        #  OrderedDict([('sampler_params',
                        #                ['1e-3,100', '1e-3,50']
                        #                )])),
                        # ('exp_snooze_stepdec',
                        #  OrderedDict([('sampler_params',
                        #                # ['1e-3,50']
                        #                ['1e-3,100']
                        #                )])),
                        # ('exp_snooze_expinc',
                        #  OrderedDict([('sampler_params',
                        #                ['1e-3,100']
                        #                )])),
                        # ('exp_snooze_stepinc',
                        #  OrderedDict([('sampler_params',
                        #                ['1e-3,100']
                        #                )])),
                    ]),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [2, 5]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', 'loss'),
                          ('dmom', [0.0, 0.9]),
                          ]+shared_args)]
    return args, log_dir


def svhn_expsnooze(args):
    # Same as cifar10 resnet32
    dataset = 'svhn'
    log_dir = 'runs_%s_expsnooze_ad_hist_hparam_nonorm' % dataset
    shared_args = [('dataset', dataset),
                   ('arch', 'resnet32'),
                   ('epochs', [
                       # (160, OrderedDict([('lr_decay_epoch', '80,120')])),
                       # (120, OrderedDict([('lr_decay_epoch', '60,90')])),
                       (160, OrderedDict([('exp_lr', ''),
                                          ('lr_decay_epoch', 80)])),
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
    args += [OrderedDict([('optim', 'sgd'),
                          ]+shared_args)]
    # args += [OrderedDict([('optim', 'dmom_jvp'),
    #                       ('alpha', 'one'),
    #                       ]+shared_args)]
    shared_args += [('optim', 'dmom_ns'),
                    # ('alpha_norm', 'sum'),
                    ('alpha_norm', 'none'),
                    ('sampler', ''),
                    ('sampler_w2c', [
                        ('linrank',
                         OrderedDict([('sampler_params',
                                       ['50,90,2,200']
                                       )])),
                        ('expsnz_tau',
                         OrderedDict([('sampler_params',
                                       [1e-3, 1e-4]
                                       )])),
                        ('expsnz_tauXstd',
                         OrderedDict([('sampler_params',
                                       [.5, .1, .01]
                                       ),
                                      ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       [.1, .05, .01]
                                       ),
                                      ])),
                    ]),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [1, 5]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', 'loss'),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.0, 0.9]),
                          ]+shared_args)]
    return args, log_dir


def imagenet_expsnooze(args):
    # from resnet https://arxiv.org/pdf/1512.03385.pdf
    dataset = 'imagenet'
    log_dir = 'runs_%s_expsnooze_ad_hist_hparam_nonorm' % dataset
    shared_args = [('dataset', dataset),
                   ('arch', 'resnet18'),
                   ('epochs', [120]),
                   ('batch_size', 256),
                   ('lr', [.1]),
                   ('workers', [12]),
                   ('weight_decay', 1e-4),
                   ('lr_decay_epoch', [60]),
                   # ('wnoise', [None, '']),
                   ('exp_lr', ['']),
                   ('train_accuracy', ['']),
                   ]
    args += [OrderedDict([('optim', 'sgd'),
                          ]+shared_args)]
    shared_args += [('optim', 'dmom_ns'),
                    # ('alpha_norm', 'sum'),
                    ('alpha_norm', 'none'),
                    ('sampler', ''),
                    ('sampler_w2c', [
                        # ('linrank',
                        #  OrderedDict([('sampler_params',
                        #                ['50,90,2,200']
                        #                )])),
                        # ('expsnz_tau',
                        #  OrderedDict([('sampler_params',
                        #                [1e-3, 1e-4]
                        #                )])),
                        # ('expsnz_tauXstd',
                        #  OrderedDict([('sampler_params',
                        #                # [.5, .1, .01]
                        #                [.1]
                        #                ),
                        #               ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [.1, .05, .01]
                                       [.1, 1]
                                       ),
                                      ])),
                    ]),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [1, 5]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('alpha', 'loss'),
                          # ('dmom', [0., 0.5, 0.9]),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_hist(args):
    dataset = 'mnist'
    # log_dir = 'runs_%s_hist' % dataset
    log_dir = 'runs_%s_bars' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   # ('lr', .02),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 100)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('arch', 'ssmlp'),
                   # ('optim', ['sgd', 'adam']),
                   ('optim', [('sgd', OrderedDict([('lr', .02)])),
                              ('adam', OrderedDict([('lr', .05)]))
                              ]),
                   ]
    args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    # ('seed', [1, 2]),
                    # ('alpha_norm', 'none'),  # default none
                    # ('alpha_norm', ['sum', 'none']),
                    # ('sampler', ''),  # True if scheduler
                    ('sampler_w2c', [
                        ('linrank',
                         OrderedDict([('sampler_params',
                                       ['50,90,2,200']
                                       )])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [1., .1, .01]
                                       [.01]
                                       ),
                                      ])),
                    ]),
                    ]
    args += [OrderedDict([('dmom', 0.9),
                          # ('alpha', 'loss'),  # default for non dmom_ns
                          # ('dmom', [0., 0.9]),
                          ]+shared_args)]
    return args, log_dir


def imagenet_hist(args):
    # from resnet https://arxiv.org/pdf/1512.03385.pdf
    dataset = 'imagenet'  # 'imagenet'
    log_dir = 'runs_%s_hist' % dataset
    shared_args = [('dataset', dataset),
                   ('optim', 'sgd'),  # 'sgd', 'adam'
                   # ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   ('arch', 'resnet50'),
                   # ('epochs', [120]),
                   ('batch_size', 256),
                   # ('lr', [.1]),
                   ('workers', [12]),
                   ('weight_decay', 1e-4),
                   # ('lr_decay_epoch', [60]),
                   # ('wnoise', [None, '']),
                   # ('exp_lr', ['']),
                   #  ### pretrained
                   ('train_accuracy', ['']),
                   # ('pretrained', ['']),
                   ('epochs', [10]),
                   ('lr', [.001]),
                   ('lr_decay_epoch', [10]),
                   ('exp_lr', [None]),
                   ]
    args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    # ('alpha_norm', 'sum'),
                    # ('alpha_norm', 'none'),
                    # ('sampler', ''),
                    ('sampler_w2c', [
                        ('linrank',
                         OrderedDict([('sampler_params',
                                       ['50,90,2,200']
                                       )])),
                        # ('expsnz_tau',
                        #  OrderedDict([('sampler_params',
                        #                [1e-3, 1e-4]
                        #                )])),
                        # ('expsnz_tauXstd',
                        #  OrderedDict([('sampler_params',
                        #                # [.5, .1, .01]
                        #                [.1]
                        #                ),
                        #               ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [.1, .05, .01]
                                       # [.1, 1]
                                       [.01]
                                       ),
                                      ])),
                    ]),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [1, 5]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('dmom', [0.9]),
                          # ('alpha', 'loss'),
                          # ('dmom', [0., 0.5, 0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_hist_corrupt(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_hist_corrupt' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   # ('lr', .02),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 100)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('arch', 'ssmlp'),
                   # ('optim', ['sgd', 'adam']),
                   ('optim', [('sgd', OrderedDict([('lr', .02)])),
                              # ('adam', OrderedDict([('lr', .05)]))
                              ]),
                   ('corrupt_perc', [None, 20, 50, 100]),
                   ('weight_decay', 0),
                   ('nodropout', [None, '']),
                   ]
    args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    # ('seed', [1, 2]),
                    # ('alpha_norm', 'none'),  # default none
                    # ('alpha_norm', ['sum', 'none']),
                    # ('sampler', ''),  # True if scheduler
                    ('sampler_w2c', [
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [1., .1, .01]
                                       [.01]
                                       ),
                                      ])),
                    ]),
                    ]
    args += [OrderedDict([('dmom', 0.9),
                          # ('alpha', 'loss'),  # default for non dmom_ns
                          # ('dmom', [0., 0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_hist_scale(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_hist_scale' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   # ('lr', .02),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 100)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('arch', 'ssmlp'),
                   # ('optim', ['sgd', 'adam']),
                   ('optim', [('sgd', OrderedDict([('lr', .02)])),
                              # ('adam', OrderedDict([('lr', .05)]))
                              ]),
                   # ('corrupt_perc', [None, 20, 50, 100]),
                   # ('weight_decay', 0),
                   # ('nodropout', [None, '']),
                   ]
    args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    # ('seed', [1, 2]),
                    # ('alpha_norm', 'none'),  # default none
                    # ('alpha_norm', ['sum', 'none']),
                    # ('sampler', ''),  # True if scheduler
                    ('sampler_w2c', [
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [1., .1, .01]
                                       [.01]
                                       ),
                                      ])),
                        ('expsnz_tauXstdLB',
                         OrderedDict([('sampler_params',
                                       [1., .1, .01]
                                       # [.01]
                                       ),
                                      ])),
                    ]),
                    ]
    args += [OrderedDict([  # ('dmom', 0.9),
                          # ('alpha', 'loss'),  # default for non dmom_ns
                          ('dmom', [0., 0.9]),
                          ]+shared_args)]
    return args, log_dir


def imagenet10_hist(args):
    # from resnet https://arxiv.org/pdf/1512.03385.pdf
    # dataset = 'imagenet10'  # 'imagenet'
    # dataset = 'imagenet10_catdog'  # 'imagenet'
    # dataset = 'imagenet5'  # 'imagenet'
    dataset = 'imagenet5_dog'  # 'imagenet'
    # dataset = 'imagenet100'  # 'imagenet'
    log_dir = 'runs_%s_hist' % dataset
    shared_args = [('dataset', dataset),
                   ('optim', 'sgd'),  # 'sgd', 'adam'
                   ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   # ('arch', 'resnet50'),
                   ('epochs', [120]),
                   ('batch_size', 256),
                   ('lr', [.01]),
                   ('workers', [12]),
                   ('weight_decay', 1e-4),
                   ('lr_decay_epoch', [60]),
                   # ('wnoise', [None, '']),
                   ('exp_lr', ['']),
                   #  ### pretrained
                   ('train_accuracy', ['']),
                   ]
    args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    # ('alpha_norm', 'sum'),
                    # ('alpha_norm', 'none'),
                    # ('sampler', ''),
                    ('sampler_w2c', [
                        ('linrank',
                         OrderedDict([('sampler_params',
                                       ['10,90,2,200']
                                       )])),
                        # ('expsnz_tau',
                        #  OrderedDict([('sampler_params',
                        #                [1e-3, 1e-4]
                        #                )])),
                        # ('expsnz_tauXstd',
                        #  OrderedDict([('sampler_params',
                        #                # [.5, .1, .01]
                        #                [.1]
                        #                ),
                        #               ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [.1, .05, .01]
                                       # [.1, 1]
                                       [.01]
                                       ),
                                      ])),
                        ('expsnz_tauXstdLB',
                         OrderedDict([('sampler_params',
                                       # [.1, .05, .01]
                                       # [.1, 1]
                                       [.01]
                                       ),
                                      ])),
                    ]),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [1, 5]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('dmom', [0.9]),
                          # ('alpha', 'loss'),
                          # ('dmom', [0., 0.5, 0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_minvar(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_minvar' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   ('lr', .02),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 100)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('arch', 'ssmlp'),
                   # ('optim', ['sgd', 'adam']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            # ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   ]
    # args += [OrderedDict([('optim', 'sgd')]+shared_args)]
    # args_1 = [('scheduler', ''),
    #           ('optim', 'sgd'),
    #           # ('seed', [1, 2]),
    #           # ('alpha_norm', 'none'),  # default none
    #           # ('alpha_norm', ['sum', 'none']),
    #           # ('sampler', ''),  # True if scheduler
    #           ('sampler_w2c', [
    #               ('linrank',
    #                OrderedDict([('sampler_params',
    #                              ['50,90,2,200']
    #                              )])),
    #               ('expsnz_tauXstdL',
    #                OrderedDict([('sampler_params',
    #                              # [1., .1, .01]
    #                              [.01]
    #                              ),
    #                             ])),
    #           ]),
    #           ]
    # args += [OrderedDict([('dmom', 0.9),
    #                       # ('alpha', 'loss'),  # default for non dmom_ns
    #                       # ('dmom', [0., 0.9]),
    #                       ]+shared_args+args_1)]
    args_2 = [('sampler', ''),
              ('optim', 'dmom'),
              ('alpha', 'normg'),
              ('minvar', ''),
              ('sampler_maxw', [2, 100]),
              # ('sampler_maxw', [100]),
              ('sampler_params', [0, 1., .1])
              ]
    args += [OrderedDict([('dmom', 0.0),
                          ]+shared_args+args_2)]
    args_2 = [('sampler', ''),
              ('optim', 'dmom'),
              ('alpha', 'rand'),
              ('minvar', ''),
              # ('sampler_maxw', [2, 100]),
              # ('sampler_maxw', [100]),
              ('sampler_params', ['.5,.5'])
              ]
    args += [OrderedDict([('dmom', 0.0),
                          ]+shared_args+args_2)]
    return args, log_dir


def cifar10_wresnet(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_wresnet' % dataset
    shared_args = [('dataset', dataset),
                   # ('arch', [
                   #     # ('resnet32', OrderedDict([
                   #     #     ('exp_lr', ''),
                   #     #     ('weight_decay', 1e-4),
                   #     #     ('lr_decay_epoch', 100)])),
                   #     ('wrn28-10', OrderedDict([
                   #         # ('exp_lr', ''),
                   #         ])),
                   # ]),
                   # ('arch', ['resnet32', 'wrn28-10']),
                   ('arch', ['wrn10-4']),
                   ('epochs', 200),
                   ('lr', 0.1),
                   ('weight_decay', 5e-4),
                   ('nesterov', ''),
                   # ('data_aug', ''),
                   # ('wnoise', [None, '']),
                   ('wnoise', ''),
                   ('optim', 'sgd'),
                   ('exp_lr', [
                       # (None, OrderedDict([
                       #     ('lr_decay_epoch', '60,120,160'),
                       #     ('lr_decay_rate', 0.2),
                       # ])),
                       ('', OrderedDict([('lr_decay_epoch', 100)])),
                   ]),
                   ]
    args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    ('sampler_w2c', [
                        ('linrank',
                         OrderedDict([('sampler_params',
                                       ['50,90,2,200']
                                       )])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [.1, .01]
                                       [.01]
                                       ),
                                      ])),
                    ]),
                    ]
    args += [OrderedDict([('alpha', 'loss'),
                          ('dmom', [0.9]),
                          ]+shared_args)]
    return args, log_dir


def imagenet_small(args):
    # from resnet https://arxiv.org/pdf/1512.03385.pdf
    dataset = 'imagenet'  # 'imagenet'
    log_dir = 'runs_%s_small' % dataset
    shared_args = [('dataset', dataset),
                   ('optim', 'sgd'),  # 'sgd', 'adam'
                   # ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   ('arch', 'alexnet'),
                   # ('epochs', [120]),
                   ('batch_size', 128),
                   # ('lr', [.1]),
                   ('workers', [12]),
                   ('weight_decay', 5e-4),
                   # ('lr_decay_epoch', [60]),
                   # ('wnoise', [None, '']),
                   # ('exp_lr', ['']),
                   #  ### pretrained
                   ('train_accuracy', ['']),
                   ('pretrained', ['']),
                   ('epochs', [20]),
                   ('lr', [.00001]),
                   ('lr_decay_epoch', [10]),
                   ('exp_lr', ['']),
                   ]
    args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    # ('alpha_norm', 'sum'),
                    # ('alpha_norm', 'none'),
                    # ('sampler', ''),
                    ('sampler_w2c', [
                        # ('linrank',
                        #  OrderedDict([('sampler_params',
                        #                ['50,90,2,200']
                        #                )])),
                        # ('expsnz_tau',
                        #  OrderedDict([('sampler_params',
                        #                [1e-3, 1e-4]
                        #                )])),
                        # ('expsnz_tauXstd',
                        #  OrderedDict([('sampler_params',
                        #                # [.5, .1, .01]
                        #                [.1]
                        #                ),
                        #               ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       [.1, .05, .01]
                                       # [.1, 1]
                                       # [.01]
                                       ),
                                      ])),
                    ]),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [1, 5]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('dmom', [0.9]),
                          ('alpha', 'loss'),
                          # ('dmom', [0., 0.5, 0.9]),
                          ]+shared_args)]
    return args, log_dir


def mnist_genlztn(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_genbig' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   ('lr', .02),
                   ('epochs', [
                       (100, OrderedDict([('lr_decay_epoch', 100)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('arch', ['cnn', 'mlp', 'ssmlp']),
                   # ('arch', 'cnn'),
                   ('arch', 'bigcnn'),
                   # ('optim', ['sgd', 'adam']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            # ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   ('log_stats', ''),
                   ('test_batch_size', 10),
                   # ('corrupt_perc', [20, 50, 100]),
                   ('corrupt_perc', [0, 100]),
                   # ('label_smoothing', [0.0, 0.1]),
                   # ('nodropout', [None, '']),
                   ]
    # args += [OrderedDict(shared_args)]
    # args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0)])]
    # args += [OrderedDict(shared_args+[('nodropout', '')])]
    # args += [OrderedDict(shared_args+[('wd', 0)])]
    args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0),
                                      ('label_smoothing', 0.1)])]
    args += [OrderedDict(shared_args+[('wnoise', '')])]
    args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0),
                                      ('wnoise', '')])]
    # args_1 = [('scheduler', ''),
    #           # ('seed', [1, 2]),
    #           # ('alpha_norm', 'none'),  # default none
    #           # ('alpha_norm', ['sum', 'none']),
    #           # ('sampler', ''),  # True if scheduler
    #           ('sampler_w2c', [
    #               # ('linrank',
    #               #  OrderedDict([('sampler_params',
    #               #                ['50,90,2,200']
    #               #                )])),
    #               ('expsnz_tauXstdL',
    #                OrderedDict([('sampler_params',
    #                              # [1., .1, .01]
    #                              [.01]
    #                              ),
    #                             ])),
    #           ]),
    #           ]
    # args += [OrderedDict([('dmom', 0.9),
    #                       # ('alpha', 'loss'),  # default for non dmom_ns
    #                       # ('dmom', [0., 0.9]),
    #                       ]+shared_args+args_1)]
    # args_2 = [('sampler', ''),
    #           ('optim', 'dmom'),
    #           ('alpha', 'normg'),
    #           ('minvar', ''),
    #           ('sampler_maxw', [2, 100]),
    #           # ('sampler_maxw', [100]),
    #           ('sampler_params', [0, 1., .1])
    #           ]
    # args += [OrderedDict([('dmom', 0.0),
    #                       ]+shared_args+args_2)]
    return args, log_dir


def mnist_duplicate(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_dups' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   ('lr', .02),
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', 100)])),
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('arch', ['cnn', 'mlp', 'ssmlp']),
                   ('arch', 'cnn'),
                   # ('arch', 'bigcnn'),
                   # ('optim', ['sgd', 'adam']),
                   ('optim', ['sgd']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            # ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   ('log_stats', ''),
                   ('test_batch_size', 10),
                   # ('corrupt_perc', [20, 50, 100]),
                   # ('corrupt_perc', [0, 100]),
                   # ('label_smoothing', [0.0, 0.1]),
                   # ('nodropout', [None, '']),
                   ('duplicate', [None, '10,10000']),
                   ]
    args += [OrderedDict(shared_args)]
    # args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0)])]
    # args += [OrderedDict(shared_args+[('nodropout', '')])]
    # args += [OrderedDict(shared_args+[('wd', 0)])]
    # args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0),
    #                                   ('label_smoothing', 0.1)])]
    args += [OrderedDict(shared_args+[('wnoise', '')])]
    # args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0),
    #                                   ('wnoise', '')])]
    args_1 = [('scheduler', ''),
              # ('seed', [1, 2]),
              # ('alpha_norm', 'none'),  # default none
              # ('alpha_norm', ['sum', 'none']),
              # ('sampler', ''),  # True if scheduler
              ('sampler_w2c', [
                  # ('linrank',
                  #  OrderedDict([('sampler_params',
                  #                ['50,90,2,200']
                  #                )])),
                  ('expsnz_tauXstdL',
                   OrderedDict([('sampler_params',
                                 # [1., .1, .01]
                                 [.01]
                                 ),
                                ])),
              ]),
              ]
    args += [OrderedDict([('dmom', 0.9),
                          # ('alpha', 'loss'),  # default for non dmom_ns
                          # ('dmom', [0., 0.9]),
                          ]+shared_args+args_1)]
    # args_2 = [('sampler', ''),
    #           ('optim', 'dmom'),
    #           ('alpha', 'normg'),
    #           ('minvar', ''),
    #           ('sampler_maxw', [2, 100]),
    #           # ('sampler_maxw', [100]),
    #           ('sampler_params', [0, 1., .1])
    #           ]
    # args += [OrderedDict([('dmom', 0.0),
    #                       ]+shared_args+args_2)]
    return args, log_dir


def mnist_diff(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_diff' % dataset
    shared_args = [('dataset', dataset),
                   # ('lr', [.1, .05, .02, .01]),
                   ('lr', .02),
                   ('epochs', [
                       # (100, OrderedDict([('lr_decay_epoch', 100)])),
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   # ('exp_lr', ''),
                   # ('wnoise', ''),
                   # ('arch', ['cnn', 'mlp', 'ssmlp']),
                   ('arch', 'cnn'),
                   # ('arch', 'bigcnn'),
                   # ('optim', ['sgd', 'adam']),
                   ('optim', ['sgd']),
                   # ('optim', [('sgd', OrderedDict([('lr', .02)])),
                   #            # ('adam', OrderedDict([('lr', .05)]))
                   #            ]),
                   ('log_stats', ''),
                   ('test_batch_size', 10),
                   # ('corrupt_perc', [20, 50, 100]),
                   # ('corrupt_perc', [0, 100]),
                   # ('label_smoothing', [0.0, 0.1]),
                   # ('nodropout', [None, '']),
                   # ('duplicate', [None, '10,10000']),
                   ]
    args += [OrderedDict(shared_args)]
    # args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0)])]
    # args += [OrderedDict(shared_args+[('nodropout', '')])]
    # args += [OrderedDict(shared_args+[('wd', 0)])]
    # args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0),
    #                                   ('label_smoothing', 0.1)])]
    # args += [OrderedDict(shared_args+[('wnoise', '')])]
    # args += [OrderedDict(shared_args+[('nodropout', ''), ('wd', 0),
    #                                   ('wnoise', '')])]
    args_1 = [('scheduler', ''),
              ('alpha_diff', ''),
              # ('seed', [1, 2]),
              # ('alpha_norm', 'none'),  # default none
              # ('alpha_norm', ['sum', 'none']),
              # ('sampler', ''),  # True if scheduler
              ('sampler_w2c', [
                  # ('linrank',
                  #  OrderedDict([('sampler_params',
                  #                ['50,90,2,200']
                  #                )])),
                  ('expsnz_tauXstdL',
                   OrderedDict([('sampler_params',
                                 [1., .1, .01]
                                 # [.01]
                                 ),
                                ])),
              ]),
              ]
    args += [OrderedDict([('dmom', [0.0, 0.5, 0.9]),
                          # ('alpha', 'loss'),  # default for non dmom_ns
                          # ('dmom', [0., 0.9]),
                          ]+shared_args+args_1)]
    # args_2 = [('sampler', ''),
    #           ('optim', 'dmom'),
    #           ('alpha', 'normg'),
    #           ('minvar', ''),
    #           ('sampler_maxw', [2, 100]),
    #           # ('sampler_maxw', [100]),
    #           ('sampler_params', [0, 1., .1])
    #           ]
    # args += [OrderedDict([('dmom', 0.0),
    #                       ]+shared_args+args_2)]
    return args, log_dir


def imagenet_diff(args):
    # from resnet https://arxiv.org/pdf/1512.03385.pdf
    dataset = 'imagenet'  # 'imagenet'
    log_dir = 'runs_%s_diff' % dataset
    shared_args = [('dataset', dataset),
                   ('optim', 'sgd'),  # 'sgd', 'adam'
                   ('arch', 'resnet18'),
                   # ('arch', 'resnet34'),
                   # ('arch', 'alexnet'),
                   # ('epochs', [120]),
                   ('batch_size', 256),
                   # ('lr', [.1]),
                   ('workers', [12]),
                   ('weight_decay', 1e-4),
                   # ('lr_decay_epoch', [60]),
                   # ('wnoise', [None, '']),
                   # ('exp_lr', ['']),
                   #  ### pretrained
                   ('train_accuracy', ['']),
                   ('pretrained', ['']),
                   ('epochs', [20]),
                   ('lr', [1e-3]),
                   ('lr_decay_epoch', [10]),
                   ('exp_lr', ['']),
                   ]
    # args += [OrderedDict(shared_args)]
    shared_args += [('scheduler', ''),
                    ('alpha_diff', ''),
                    # ('alpha_norm', 'sum'),
                    # ('alpha_norm', 'none'),
                    # ('sampler', ''),
                    ('sampler_w2c', [
                        # ('linrank',
                        #  OrderedDict([('sampler_params',
                        #                ['50,90,2,200']
                        #                )])),
                        # ('expsnz_tau',
                        #  OrderedDict([('sampler_params',
                        #                [1e-3, 1e-4]
                        #                )])),
                        # ('expsnz_tauXstd',
                        #  OrderedDict([('sampler_params',
                        #                # [.5, .1, .01]
                        #                [.1]
                        #                ),
                        #               ])),
                        ('expsnz_tauXstdL',
                         OrderedDict([('sampler_params',
                                       # [.1, .05, .01]
                                       # [.1, 1]
                                       [.01]
                                       ),
                                      ])),
                    ]),
                    # ('seed', [1, 2]),
                    # ('sampler_maxw', [1, 5]),
                    ]
    # args += [OrderedDict([('alpha', 'one'),
    #                       ]+shared_args)]
    args += [OrderedDict([('dmom', [0.0, 0.5, 0.9]),
                          # ('alpha', 'loss'),
                          # ('dmom', [0., 0.5, 0.9]),
                          ]+shared_args)]
    return args, log_dir


def main():
    # module_name ='zerol.main'
    # args, log_dir = cifar10_resnet_explr(args)
    # args, log_dir = mnist_explr(args)
    # args, log_dir = svhn_ggbar(args)
    # args, log_dir = imagenet(args)
    # args, log_dir = mnist_linear2(args)
    # args, log_dir = cifar10_linear2(args)
    # args, log_dir = svhn_linear2(args)
    # args, log_dir = imagenet_linear2_bs(args)
    # args, log_dir = mnist_bs(args)
    # args, log_dir = cifar10_bs(args)
    # args, log_dir = svhn_bs(args)
    # args, log_dir = mnist_autoexp(args)
    # args, log_dir = cifar10_autoexp(args)
    # args, log_dir = mnist_linrank(args)
    # args, log_dir = cifar10_linrank(args)
    # args, log_dir = svhn_linrank(args)
    # args, log_dir = mnist_expsnooze(args)
    # args, log_dir = cifar10_expsnooze(args)
    # args, log_dir = svhn_expsnooze(args)
    # args, log_dir = imagenet_expsnooze(args)
    # args, log_dir = mnist_hist(args)
    # args, log_dir = imagenet_hist(args)
    # args, log_dir = imagenet10_hist(args)
    # args, log_dir = mnist_hist_corrupt(args)
    # args, log_dir = mnist_hist_scale(args)
    # args, log_dir = mnist_minvar(args)
    # args, log_dir = cifar10_wresnet(args)
    # args, log_dir = imagenet_small(args)
    # args, log_dir = mnist_genlztn(args)
    # args, log_dir = mnist_duplicate(args)
    # args, log_dir = mnist_diff(args)
    # args, log_dir = imagenet_diff(args)
    ret = []
    return ret
