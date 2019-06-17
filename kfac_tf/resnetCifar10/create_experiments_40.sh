#!/bin/bash
fileName='cifar10_main.py'
optimizer='Adam'
resnet_size=32
model_dir='./logs/ResNet'$resnet_size
epochs='250 500 1000 2000'
batch_size='128 256 512 1024'
lr='0.0005 0.0005 0.0005 0.0005'
Lambda='0.1'
noiseType='DiagF'
eps='1e-8'
# If you want to resume the training at epoch 5 for a given configuration
# Uncomment the line below
#resume='/checkpoint_1.tar'
# and add in output --resume $results_dir$model$dataset$optimizer/$save${bs[j-1]}$resume

exp_num='40'
j=1
for epoch in $epochs
do
    bs=($batch_size)
    curr_lr=($lr)
    header="#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=2\n#SBATCH --partition=gpuc\n#SBATCH --exclude guppy8,guppy9,guppy10,guppy13,guppy20\n\n"
    printf "$header" > exp${exp_num}_$j.sh
    output="python $fileName --optimizer $optimizer \
	--data_dir \$HOME/datasets/cifar-10-batches-bin \
        --model_dir $model_dir\
	--train_epochs $epoch\
	--epochs_per_eval 1\
        --batch_size ${bs[j-1]}\
	 --lr ${curr_lr[j-1]} --noiseLambda $Lambda --useNoiseType $noiseType --eps $eps"
    printf "$output">> exp${exp_num}_$j.sh
    if [ $saveExp ];
    then
        printf "$output\n\n" >> save_experiments.txt
    fi
    ((j++))
done





