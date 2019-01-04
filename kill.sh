#!/bin/bash
machine=$1
pattern=$2

ssh bolt$machine \
    ps -ef \| grep \"jobs.*.sh\"  \| awk \'{print '$2'}\' \| xargs kill -9
ssh bolt$machine \
    ps -ef \| grep \"python -m main\"  \| grep \"$pattern\" \| awk \'{print '$2'}\' \| xargs kill -9
