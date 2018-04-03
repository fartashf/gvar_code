from __future__ import print_function
from collections import OrderedDict
from itertools import product


class RunSingle(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.num = 0

    def __call__(self, args):
        cmd = ['python main.py']
        logger_name = '%s/%d_' % (self.log_dir, self.num)
        self.num += 1
        for k, v in args:
            cmd += ['--{} {}'.format(k, v)]
            logger_name += '{}_{},'.format(k, v)
        cmd += ['--logger_name '+logger_name.strip(',')]
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
    args += [OrderedDict([('dataset', ['mnist']),
                          ('optim', ['dmom']),
                          ('lr', [0.01]),
                          ('dmom', [0.]),
                          ('momentum', [0.9]),
                          ('alpha', ['one']),
                          ('batch_size', [12, 13]),
                          ])]
    log_dir = 'runs_mnist'
    njobs = 3

    jobs_0 = ['bolt1_gpu0', 'bolt1_gpu1', 'bolt1_gpu2', 'bolt1_gpu3',
              'bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3',
              'bolt0_gpu0', 'bolt0_gpu1', 'bolt0_gpu2', 'bolt0_gpu3']
    jobs = []
    for i in range(njobs):
        jobs += ['%s_job%d' % (s, i) for s in jobs_0]

    run_single = RunSingle(log_dir)
    run_single.num = 29

    cmds = run_multi(run_single, args)
    print(len(cmds))
    for j, job in enumerate(jobs):
        with open('jobs/{}.sh'.format(job), 'w') as f:
            for i in range(j, len(cmds), len(jobs)):
                print(cmds[i], file=f)
