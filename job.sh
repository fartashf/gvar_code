#!/bin/bash
machine=$1
gpu=$2
job=$3

ssh bolt$machine \
    source /ais/fleet10/faghri/export2.sh\; \
    cd $HOME/dmom/code/\;\
    CUDA_VISIBLE_DEVICES=$gpu sh jobs/bolt"$machine"_gpu"$gpu"_job"$job".sh
