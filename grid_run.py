from __future__ import print_function
from collections import OrderedDict
from itertools import product


class RunSingle(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.num = 0

    def __call__(self, args):
        logger_name = '%s/%d_' % (self.log_dir, self.num)
        cmd = ['python main.py']
        self.num += 1
        for k, v in args:
            cmd += ['--{} {}'.format(k, v)]
            logger_name += '{}_{},'.format(k, v)
        dir_name = logger_name.strip(',')
        cmd += ['--logger_name "$dir_name"']
        cmd += ['> "$dir_name/log" 2>&1']
        cmd = ['dir_name="%s"; mkdir -p "$dir_name" && ' % dir_name] + cmd
        return ' '.join(cmd)


def run_multi(run_single, args):
    cmds = []
    if isinstance(args, list):
        for a in args:
            cmds += run_multi(run_single, a)
    elif isinstance(args, dict):
        keys = args.keys()
        values = args.values()
        for v in product(*values):
            p = zip(keys, v)
            cmds += [run_single(p)]
    else:
        cmds = run_single(args)
    return cmds


if __name__ == '__main__':
    # args = [OrderedDict([('optim', ['sgd']),
    #                      ('lr', [0.1, 0.01, 0.001]),
    #                      ('momentum', [0.1, 0.5, 0.9, 0.99]),
    #                      ]),
    #         OrderedDict([('optim', ['dmom']),
    #                      ('lr', [0.1, 0.01, 0.001]),
    #                      ('dmom', [0.1, 0.5, 0.9, 0.99]),
    #                      ('momentum', [0.1, 0.5, 0.9, 0.99]),
    #                      ])]
    args = []
    # sgd
    # args += [OrderedDict([('optim', ['dmom']),
    #                       ('lr', [0.1, 0.01, 0.001]),
    #                       ('dmom', [0]),
    #                       ('momentum', [0, 0.5, 0.9, 0.99]),
    #                       ])]
    # args += [OrderedDict([('optim', ['dmom']),
    #                       ('lr', [0.1, 0.01, 0.001]),
    #                       ('dmom', [0.5, 0.9, 0.99]),
    #                       ('momentum', [0., 0.5, 0.9, 0.95, 0.99]),
    #                       # ('dmom_interval', [1, 3, 5]),
    #                       # ('dmom_temp', [0, 1, 2]),
    #                       # ('alpha', ['jacobian']),
    #                       # ('jacobian_lambda', [
    #                       # ('dmom_theta', [0, 10]),
    #                       ('low_theta', [1e-4, 1e-2]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['logreg']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['ggbar_abs', 'ggbar_max0']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [.01]),
    #                       ('sampler_alpha_th', [.00000001, .000001, .00001]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['normg']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [1]),
    #                       ('sampler_alpha_th', [.00000001, .000001, .00001]),
    #                       ])]
    # ##############
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.001]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('batch_size', [12, 13]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       # ('dmom_interval', [1, 3, 5]),
    #                       # ('dmom_temp', [0, 1, 2]),
    #                       ('alpha', ['normg',
    #                                  'ggbar_abs', 'ggbar_max0', 'loss']),
    #                       # ('jacobian_lambda', [
    #                       # ('dmom_theta', [0, 10]),
    #                       # ('low_theta', [1e-4, 1e-2]),
    #                       ('alpha_norm', ['none', 'sum']),
    #                       # 'sum_batch', 'exp_batch']),
    #                       # ('weight_decay', [0, 5e-4]),
    #                       # ('wmomentum', [False, True]),
    #                       # ('arch', ['mlp', 'convnet']),
    #                       # ('norm_temp', [1, 2, 10, 100]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['normg', 'ggbar_abs', 'ggbar_max0',
    #                                  'loss']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [1, .01]),
    #                       ])]
    ############################
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['normg']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [1]),
    #                       ('sampler_alpha_perc', [30, 50, 70, 90]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['ggbar_abs', 'ggbar_max0']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [.01]),
    #                       ('sampler_alpha_perc', [30, 50, 70, 90]),
    #                       ])]
    # # sampler hyper param search
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['normg']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [1]),
    #                       ('sampler', ['']),
    #                       ('sampler_weight_to_count', ['1_over', 'log']),
    #                       ('sampler_max_count', [10, 20, 50]),
    #                       ('sampler_start_epoch', [1, 3, 10]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['ggbar_abs']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [.01]),
    #                       ('sampler', ['']),
    #                       ('sampler_weight_to_count', ['1_over', 'log']),
    #                       ('sampler_max_count', [10, 20, 50]),
    #                       ('sampler_start_epoch', [1, 3, 10]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['normg']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [1]),
    #                       ('sampler', ['']),
    #                       ('sampler_weight_to_count', ['1_over']),
    #                       ('sampler_max_count', [50]),
    #                       ('sampler_start_epoch', [1]),
    #                       ('sampler_lr_update', ['']),
    #                       ('sampler_lr_window', [20, 40, 100]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['ggbar_abs', 'normg']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [.01]),
    #                       ('sampler', ['']),
    #                       ('sampler_weight_to_count', ['1_over']),
    #                       ('sampler_max_count', [1000, 10000]),
    #                       ('sampler_start_epoch', [1]),
    #                       ('sampler_lr_update', ['']),
    #                       ('sampler_lr_window', [20, 40]),
    #                       ])]
    # # logreg figure
    # args += [OrderedDict([('dataset', ['logreg']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['normg', 'ggbar_abs']),
    #                       # ('alpha_norm', ['exp']),
    #                       # ('norm_temp', [.0001]),
    #                       ('alpha_norm', ['sum', 'sum_class']),
    #                       ])]
    # ##### /len
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['exp', 'none']),
    #                       ('norm_temp', [1]),
    #                       ('divlen', [0, 1]),
    #                       ('sampler_alpha_perc', [0, 90]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['normg']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [1]),
    #                       ('divlen', [0, 1]),
    #                       ('sampler_alpha_perc', [0, 90]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['ggbar_abs']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [.01]),
    #                       ('divlen', [0, 1]),
    #                       ('sampler_alpha_perc', [0, 90]),
    #                       ])]
    # wmomentum fixed
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['exp']),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0., 0.5, 0.9]),
    #                       ('momentum', [0.9]),
    #                       ('wmomentum', [0., 0.5, 0.9]),
    #                       ('alpha', ['normg', 'ggbar_abs']),
    #                       ('alpha_norm', ['exp']),
    #                       ('norm_temp', [100, 10, 1, .1]),
    #                       ])]
    # sgd equivalent
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['sgd']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ])]
    # args += [OrderedDict([('dataset', ['mnist']),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('wmomentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['exp', 'sum', 'none']),
    #                       ])]
    # sampler
    # dataset = 'mnist'
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['sgd']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('wmomentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['none']),
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['none']),
    #                       ('sampler', [''])
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('momentum', [0.9]),
    #                       ('alpha_norm', ['sum']),
    #                       # ('alpha_norm', ['none']),
    #                       ('sampler', ['']),
    #                       ('alpha', ['ggbar_abs', 'normg', 'loss']),  #
    #                       ('dmom', [0., 0.5, 0.9]),
    #                       # ('wmomentum', [0., 0.5, 0.9]),
    #                       # ('norm_temp', [10, 1., .1]),
    #                       ('sampler_weight_to_count', ['linear']),
    #                       # ('sampler_max_count', [20, 50]),
    #                       # ('sampler_start_epoch', [1, 2]),
    #                       ('sampler_linear_params',
    #                        ['50,90,20,100', '50,99,20,100', '50,90,40,100']),
    #                       ])]
    # dataset = 'cifar10'
    # log_dir = 'runs_%s_sampler' % dataset
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['sgd']),
    #                       ('lr', [0.01]),
    #                       ('momentum', [0.9]),
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('wmomentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['none']),
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['none']),
    #                       ('sampler', [''])
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom']),
    #                       ('lr', [0.01]),
    #                       ('momentum', [0.9]),
    #                       ('alpha_norm', ['sum']),
    #                       # ('alpha_norm', ['none']),
    #                       ('sampler', ['']),
    #                       ('sampler_linear_params',
    #                        ['20,70,20,100', '20,70,40,100']),
    #                       ('alpha', ['ggbar_abs', 'normg', 'loss']),  #
    #                       ('dmom', [0., 0.5, 0.9]),
    #                       # ('wmomentum', [0., 0.5, 0.9]),
    #                       # ('norm_temp', [10, 1., .1]),
    #                       ('sampler_weight_to_count', ['linear']),
    #                       # ('sampler_max_count', [20, 50]),
    #                       # ('sampler_start_epoch', [1, 2]),
    #                       ])]
    # dataset = 'cifar10'
    # log_dir = 'runs_%s_jvp' % dataset
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['sgd']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom_jvp']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('wmomentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['none']),
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom_jvp']),
    #                       ('lr', [0.01]),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ('alpha', ['one']),
    #                       ('alpha_norm', ['none']),
    #                       ('sampler', [''])
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom_jvp']),
    #                       ('seed', [1, 2]),
    #                       ('lr', [0.01]),
    #                       ('momentum', [0.9]),
    #                       ('alpha_norm', ['sum']),
    #                       ('sampler', ['']),
    #                       ('alpha', ['one']),
    #                       ('sampler_weight_to_count', ['linear']),
    #                       ('sampler_linear_params', ['20,70,20,100']),
    #                       ])]
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('optim', ['dmom_jvp']),
    #                       ('seed', [1, 2]),
    #                       ('lr', [0.01]),
    #                       ('momentum', [0.9]),
    #                       ('alpha_norm', ['sum']),
    #                       ('sampler', ['']),
    #                       ('alpha', [  # 'loss',
    #                                  'ggbar_abs', 'loss', 'gFigbar_abs',
    #                                  'lgg', 'gtheta_abs',
    #                                  ]),
    #                       # ('alpha', ['gFgbar_abs', 'gFigbar_abs']),
    #                       ('dmom', [0., 0.5, 0.9]),
    #                       ('sampler_weight_to_count', ['linear']),
    #                       ('sampler_linear_params',
    #                        ['20,70,20,100']),
    #                       ])]
    # njobs = [3] * 8
    dataset = 'cifar10'
    log_dir = 'runs_%s_resnet' % dataset
    # args += [OrderedDict([('dataset', [dataset]),
    #                       ('arch', ['resnet32']),
    #                       ('optim', ['sgd']),
    #                       ('epochs', [200]),
    #                       ('lr', [0.1]),
    #                       ('weight_decay', [1e-4]),
    #                       ('lr_decay_epoch', ['100,150']),
    #                       ('dmom', [0.]),
    #                       ('momentum', [0.9]),
    #                       ])]
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
                          ('sampler_weight_to_count', ['linear']),
                          ('sampler_linear_params', ['20,70,50,200']),
                          ('sampler_repetition', ['']),
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
                          ('sampler_weight_to_count', ['linear']),
                          ('sampler_linear_params', ['20,70,50,200']),
                          ('sampler_repetition', ['']),
                          ])]
    njobs = [1] * 8

    jobs_0 = ['bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3',
              'bolt1_gpu0', 'bolt1_gpu1', 'bolt1_gpu2', 'bolt1_gpu3',
              ]  # 'bolt0_gpu0', 'bolt0_gpu1', 'bolt0_gpu2']  # , 'bolt0_gpu3']
    # njobs = [2, 2, 2, 2,
    #          1, 1, 1, 1]
    jobs = []
    for s, n in zip(jobs_0, njobs):
        jobs += ['%s_job%d' % (s, i) for i in range(n)]
        # jobs += ['%s_job%d' % (s, i) for s in jobs_0]

    run_single = RunSingle(log_dir)
    run_single.num = 9

    cmds = run_multi(run_single, args)
    print(len(cmds))
    for j, job in enumerate(jobs):
        with open('jobs/{}.sh'.format(job), 'w') as f:
            for i in range(j, len(cmds), len(jobs)):
                print(cmds[i], file=f)
