#! /bin/bash

fileName='exp40_'

# first job - no dependencies
j=1
j1=$(sbatch $fileName$j.sh)
jid1=${j1: -6}
echo $jid1
((j+=1))
j2=$(sbatch  --dependency=afterany:$jid1 $fileName$j.sh)
jid2=${j2: -6}
((j+=1))
jid3=$(sbatch  --dependency=afterany:$jid2 $fileName$j.sh)
# Fourth job (Batch Size = 1024) doesn't have any dependencies
((j+=1))
jid4=$(sbatch $fileName$j.sh)

# show dependencies in squeue output:
squeue -u $USER -o "%.8A %.4C %.10m %.20E"
