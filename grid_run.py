from __future__ import print_function
from collections import OrderedDict
from itertools import product


def run_single(args):
    cmd = ['python main.py']
    logger_name = 'runs_grid/'
    for k, v in args:
        cmd += ['--{} {}'.format(k, v)]
        logger_name += '{}_{},'.format(k, v)
    cmd += ['--logger_name '+logger_name.strip(',')]
    return ' '.join(cmd)


def run_multi(args):
    cmds = []
    if isinstance(args, list):
        for a in args:
            cmds += run_multi(a)
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
    args = [OrderedDict([('optim', ['dmom']),
                         ('lr', [0.1, 0.01, 0.001]),
                         ('dmom', [0]),
                         ('momentum', [0.1, 0.5, 0.9, 0.99]),
                         ])]

    jobs = ['bolt0_gpu0', 'bolt0_gpu1', 'bolt0_gpu2', 'bolt0_gpu3',
            'bolt1_gpu0', 'bolt1_gpu1', 'bolt1_gpu2', 'bolt1_gpu3',
            'bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3']

    cmds = run_multi(args)
    print(len(cmds))
    for j, job in enumerate(jobs):
        with open('jobs/{}.sh'.format(job), 'w') as f:
            for i in range(j, len(cmds), len(jobs)):
                print(cmds[i], file=f)
