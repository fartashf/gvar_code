#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --n_layer 6 \
        --d_model 256  \
        --n_head 8 \
        --d_head 64 \
        --d_inner 1024 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.003 \
        --warmup_step 0 \
        --max_step 400000 \
        --tgt_len 512 \
        --mem_len 0 \
        --attn_type 2 \
        --eval_tgt_len 128 \
        --log-interval 200 \
        --batch_size 44 \
        --g_estim ntk \
        --momentum 0.9 \
        --gvar_start 0 \
        --g_optim  \
        --g_optim_start 0 \
        --g_epoch \
        --g_estim ntk \
        --ntk_damping 0.01 \
        --ntk_cpu \
        --weight_decay 0 \
        --gvar_log_iter 200 \
        --niters 80000 \
        --lr_decay_epoch 40000,60000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
