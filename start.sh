#!/bin/sh
# for i in 0 1 2; do for j in 0 1 2 3; do for k in 0 1 2; do (nohup ./job.sh $i $j $k > jobs/log/bolt"$i"_gpu"$j"_job"$k".log 2>&1 &); done; sleep 1; done; done
for i in 1 2 3  # 0 1 2
do
    for j in 0 1 2 3 "0,1" "0,2" "0,3" "1,2" "1,3" "2,3" "0,1,2,3"
    do
        for k in 0 1 2 3 4 5 6
        do
            [ -e jobs/bolt"$i"_gpu"$j"_job"$k".sh ] && \
            (nohup ./job.sh $i $j $k > jobs/log/bolt"$i"_gpu"$j"_job"$k".log \
            2>&1 &) && sleep 1 
        done
        # sleep 1
    done
done
