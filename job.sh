#!/bin/bash
machine=$1
gpu=$2
job=$3

ssh bolt$machine \
    source /ais/fleet10/faghri/export_p1.sh\; \
    cd $HOME/nuq/\;\
    CUDA_VISIBLE_DEVICES=$gpu sh jobs/bolt"$machine"_gpu"$gpu"_job"$job".sh
