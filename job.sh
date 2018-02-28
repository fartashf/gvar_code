#!/bin/bash
machine=$1
gpu=$2

ssh bolt$machine \
    source /ais/fleet10/faghri/export.sh\; \
    cd $HOME/dmom\;\
    CUDA_VISIBLE_DEVICES=$gpu sh jobs/bolt"$machine"_gpu"$gpu".sh
