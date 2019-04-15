from __future__ import print_function


def bolt(sargs):
    """
    rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid G --run_name X
    pattern=""; for i in 1 2; do ./kill.sh $i $pattern; done
    ./start.sh
    """
    # jobs_0 = ['bolt0_gpu0,1,2,3', 'bolt1_gpu0,1,2,3']
    # jobs_0 = ['bolt2_gpu0,3', 'bolt2_gpu1,2',
    #           'bolt1_gpu0,1', 'bolt1_gpu2,3',
    #           ]
    # jobs_0 = ['bolt2_gpu0,3', 'bolt2_gpu1,2',
    #           'bolt1_gpu0,1', 'bolt1_gpu2,3',
    #           ]
    jobs_0 = ['bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3',
              'bolt1_gpu3',
              'bolt1_gpu0', 'bolt1_gpu1', 'bolt1_gpu2',
              # 'bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3',
              # 'bolt1_gpu2', 'bolt1_gpu3',
              # 'bolt1_gpu0', 'bolt1_gpu1', 'bolt1_gpu2', 'bolt1_gpu3',
              # 'bolt0_gpu0', 'bolt0_gpu1', 'bolt0_gpu2', 'bolt0_gpu3'
              ]
    # njobs = [3] * 4 + [2] * 4  # validate start.sh
    njobs = [2]*5 + [1]*3  # Number of parallel jobs on each machine
    # njobs = [2, 2, 1, 1]
    jobs = []
    for s, n in zip(jobs_0, njobs):
        jobs += ['%s_job%d' % (s, i) for i in range(n)]
        # jobs += ['%s_job%d' % (s, i) for s in jobs_0]
    parallel = False  # each script runs in sequence
    return jobs, parallel


def vector(sargs):
    """
    vector(q): gpu, wsgpu
    vaughan(vremote): p100, t4

    rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid G --run_name X \
    --cluster_args 5,4,gpu
    pattern=""; for i in 1 2; do ./kill.sh $i $pattern; done
    sbatch jobs/slurm.sbatch
    """
    njobs, ntasks, partition = sargs.split(',')
    njobs = int(njobs)
    ntasks = int(ntasks)
    # njobs = 5  # Number of array jobs
    # ntasks = 4  # Number of running jobs
    # partition = 'gpu'
    jobs = [str(i) for i in range(njobs)]
    sbatch_f = """#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=jobs/log/array_%A_%a.log
#SBATCH --array=0-{njobs}
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH -c 8
#SBATCH --mem=4G
#SBATCH -p {partition}
#SBATCH --ntasks={ntasks}

date; hostname; pwd
nvidia-smi

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
source $HOME/export_p1.sh
bash jobs/$SLURM_ARRAY_TASK_ID.sh
""".format(njobs=njobs-1, ntasks=ntasks, partition=partition)
    with open('jobs/slurm.sbatch', 'w') as f:
        print(sbatch_f, file=f)
    parallel = True  # each script runs in parallel
    return jobs, parallel
