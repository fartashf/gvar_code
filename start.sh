#!/bin/sh
for i in 0 1 2; do for j in 0 1 2 3; do for k in 0 1 2; do (nohup ./job.sh $i $j $k > jobs/log/bolt"$i"_gpu"$j"_job"$k".log 2>&1 &); done; sleep 1; done; done
