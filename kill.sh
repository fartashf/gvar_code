#!/bin/bash
machine=$1

ssh bolt$machine \
    ps -ef \| grep \"jobs.*.sh\"  \| awk \'{print '$2'}\' \| xargs kill -9
ssh bolt$machine \
    ps -ef \| grep \"python main.py\"  \| awk \'{print '$2'}\' \| xargs kill -9
