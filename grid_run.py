from __future__ import print_function
import argparse
import grid
import grid.dmom
import grid.gvar
# from itertools import product


class RunSingle(object):
    def __init__(self, log_dir, module_name, exclude):
        self.log_dir = log_dir
        self.num = 0
        self.module_name = module_name
        self.exclude = exclude

    def __call__(self, args):
        logger_name = 'runs/%s/%02d_' % (self.log_dir, self.num)
        cmd = ['python -m {}'.format(self.module_name)]
        self.num += 1
        for k, v in args:
            if v is not None:
                cmd += ['--{} {}'.format(k, v)]
                if k not in self.exclude:
                    logger_name += '{}_{},'.format(k, v)
        dir_name = logger_name.strip(',')
        cmd += ['--logger_name "$dir_name"']
        cmd += ['> "$dir_name/log" 2>&1']
        cmd = ['dir_name="%s"; mkdir -p "$dir_name" && ' % dir_name] + cmd
        return ' '.join(cmd)


def deep_product(args, index=0, cur_args=[]):
    if index >= len(args):
        yield cur_args
    elif isinstance(args, list):
        # Disjoint
        for a in args:
            for b in deep_product(a):
                yield b
    elif isinstance(args, tuple):
        # Disjoint product
        for a in deep_product(args[index]):
            next_args = cur_args + a
            for b in deep_product(args, index+1, next_args):
                yield b
    elif isinstance(args, dict):
        # Product
        # keys = args.keys()
        # values = args.values()
        # for v in product(*values):
        keys = args.keys()
        values = args.values()
        if not isinstance(values[index], list):
            values[index] = [values[index]]
        for v in values[index]:
            if not isinstance(v, tuple):
                next_args = cur_args + [(keys[index], v)]
                for a in deep_product(args, index+1, next_args):
                    yield a
            else:
                for dv in deep_product(v[1]):
                    next_args = cur_args + [(keys[index], v[0])]
                    next_args += dv
                    for a in deep_product(args, index+1, next_args):
                        yield a


def run_multi(run_single, args):
    cmds = []
    for arg in deep_product(args):
        cmds += [run_single(arg)]
    return cmds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default='', type=str)
    parser.add_argument('--run_name', default='', type=str)
    args = parser.parse_args()
    val = grid.__dict__[args.grid].__dict__[args.run_name]([])
    args, log_dir, module_name, exclude = val
    # jobs_0 = ['bolt0_gpu0,1,2,3', 'bolt1_gpu0,1,2,3']
    # jobs_0 = ['bolt2_gpu0,3', 'bolt2_gpu1,2',
    #           'bolt1_gpu0,1', 'bolt1_gpu2,3',
    #           ]
    # jobs_0 = ['bolt2_gpu0,3', 'bolt2_gpu1,2',
    #           'bolt1_gpu0,1', 'bolt1_gpu2,3',
    #           ]
    jobs_0 = ['bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3',
              'bolt1_gpu1', 'bolt1_gpu2', 'bolt1_gpu3',
              'bolt1_gpu0', 'bolt2_gpu0',
              # 'bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3',
              # 'bolt1_gpu2', 'bolt1_gpu3',
              # 'bolt1_gpu0', 'bolt1_gpu1', 'bolt1_gpu2', 'bolt1_gpu3',
              # 'bolt0_gpu0', 'bolt0_gpu1', 'bolt0_gpu2', 'bolt0_gpu3'
              ]
    # njobs = [3] * 4 + [2] * 4  # validate start.sh
    njobs = [4]*6 + [2]*2
    # njobs = [2, 2, 1, 1]
    jobs = []
    for s, n in zip(jobs_0, njobs):
        jobs += ['%s_job%d' % (s, i) for i in range(n)]
        # jobs += ['%s_job%d' % (s, i) for s in jobs_0]

    run_single = RunSingle(log_dir, module_name, exclude)
    # run_single.num = 41

    # args = OrderedDict([('lr', [1, 2]), ('batch_size', [10, 20])])
    # args = OrderedDict([('lr', [(1, OrderedDict([('batch_size', [10])])),
    #                             (2, OrderedDict([('batch_size', [20])]))])])
    # args = args[0]
    # for cmd in deep_product(args):
    #     print(cmd)

    cmds = run_multi(run_single, args)
    print(len(cmds))
    for j, job in enumerate(jobs):
        with open('jobs/{}.sh'.format(job), 'w') as f:
            for i in range(j, len(cmds), len(jobs)):
                print(cmds[i], file=f)
