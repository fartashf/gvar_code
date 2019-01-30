## Jan 32

```
CUDA_VISIBLE_DEVICES=3 python -m main.gvar --dataset mnist --lr 0.001 --epochs 30 --lr_decay_epoch 30 --arch cnn --gvar_start 0 --g_bsnap_iter 5  --g_optim_start 0 --g_epoch  --g_estim gluster --g_nclusters 10 --g_debug  --g_online  --g_osnap_iter 10 --g_beta 0.99 --g_min_size 0.001 --g_init_mul 2 --logger_name runs/mnist_adam_snap --g_save_snap  --g_noMulNk --optim adam
CUDA_VISIBLE_DEVICES=3 python -m main.gvar --dataset mnist --lr 0.05 --epochs 30 --lr_decay_epoch 30 --arch cnn --gvar_start 0 --g_bsnap_iter 5  --g_optim_start 0 --g_epoch  --g_estim gluster --g_nclusters 10 --g_debug  --g_online  --g_osnap_iter 10 --g_beta 0.99 --g_min_size 0.001 --g_init_mul 2 --logger_name runs/mnist_snap --g_save_snap  --g_noMulNk
```

## Jan 18

```
CUDA_VISIBLE_DEVICES=0 python -m main.gvar --dataset cifar10 --arch resnet20 --epochs 200 --lr_decay_epoch 100,150 --weight_decay 0.0001 --batch_size 128 --lr 0.1 --gvar_estim_iter 10 --gvar_log_iter 2000 --g_estim gluster --g_nclusters 64 --g_debug --gb_citers 5 --g_min_size 100 --gvar_start 101 --g_bsnap_iter 10 --g_optim  --g_optim_start 101 --g_epoch  --logger_name runs/cifar10_blup
CUDA_VISIBLE_DEVICES=3 python -m main.gvar --dataset cifar10 --arch resnet20 --epochs 100 --lr_decay_epoch 50 --weight_decay 0.0001 --batch_size 128 --lr 0.01 --gvar_estim_iter 10 --gvar_log_iter 1000 --g_estim sgd --g_debug --gvar_start 1 --g_epoch  --logger_name runs/runs_cifar10_blup/g_estim_sgd,epoch_100,200 --resume runs/runs_cifar10_blup/sgd,epoch_0,100/ --ckpt_name checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=0 python -m main.gvar --dataset cifar10 --arch resnet20 --epochs 100 --lr_decay_epoch 50 --weight_decay 0.0001 --batch_size 128 --lr 0.01 --gvar_estim_iter 10 --gvar_log_iter 1000 --g_estim gluster --g_nclusters 100 --g_debug --gb_citers 2 --g_min_size 100 --gvar_start 1 --g_bsnap_iter 10 --g_optim  --g_optim_start 1 --g_epoch  --logger_name runs/runs_cifar10_blup/g_estim_gluster,epoch_100,200 --resume runs/runs_cifar10_blup/sgd,epoch_0,100 --ckpt_name checkpoint.pth.tar
```
## Jan 17

```
CUDA_VISIBLE_DEVICES=0 python -m main.gvar --dataset mnist --arch cnn --lr 0.01 --gvar_start 0 --g_bsnap_iter 100000 --g_optim  --g_optim_start 1 --g_estim gluster --g_nclusters 10 --g_debug  --g_online  --g_osnap_iter 10 --g_beta 0.99 --g_min_size 0.001 --logger_name runs/Y --resume runs/X --ckpt_name checkpoint.pth.tar --g_resume
CUDA_VISIBLE_DEVICES=0 python -m main.gvar --dataset mnist --arch cnn --lr 0.1 --gvar_start 1000 --g_bsnap_iter 1000 --g_optim  --g_optim_start 1001 --g_estim gluster --g_nclusters 128 --g_debug  --g_online  --g_osnap_iter 10 --g_beta 0.99 --g_min_size 0.001 --logger_name runs/X --g_optim_max 50
```

## Dec 26
gvar
```
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main/gvar.py --dataset cifar10 --arch resnet32 --epochs 10 --lr 0.1 --weight_decay 1e-4  --logger_name runs/cifar10 --g_estim gluster --gvar_estim_iter 10 --gvar_snap_iter 400 --log_interval 10 --gvar_log_iter 10 --gb_citers 10 --g_nclusters 10 --gvar_start 800
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main/gvar.py --dataset mnist --optim sgd --lr 0.1 --momentum 0.9   --logger_name runs/mnist --epochs 10 --g_estim gluster --gvar_estim_iter 100 --gvar_snap_iter 468 --log_interval 10 --gvar_log_iter 10 --gb_citers 10 --g_nclusters 10 --gvar_start $((468*2)) --arch mlp
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main/gvar.py --dataset 10class --optim sgd --lr 0.1 --momentum 0.9   --logger_name runs/10class --epochs 10 --g_estim gluster --gvar_estim_iter 100 --gvar_snap_iter 10 --log_interval 10 --gvar_log_iter 10 --gb_citers 10 --g_nclusters 10 --gvar_start 200  --batch_size 8
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main/gvar.py --dataset mnist --optim sgd --lr 0.1 --momentum 0.9   --logger_name runs/mnist --epochs 10 --g_estim svrg --gvar_estim_iter 10 --gvar_snap_iter 10 --log_interval 10 --gvar_log_iter 10
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main/gvar.py --dataset mnist --optim sgd --lr 0.1 --momentum 0.9   --logger_name runs/mnist --epochs 10 --g_estim sgd --gvar_estim_iter 10 --gvar_snap_iter 10 --log_interval 10 --gvar_log_iter 10
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main/gvar.py --dataset 5class --optim sgd --lr 0.1 --momentum 0.9   --logger_name runs/5class --epochs 10 --g_estim svrg --gvar_estim_iter 100 --gvar_snap_iter 10 --log_interval 10 --gvar_log_iter 10 --gb_citers 10
```

## Dec 17
gluster online
```
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- bgluster.py --dataset imagenet --arch resnet34 --run_dir runs/resnet34_online --gb_citers 5000 --pretrained --test_batch_size 64 --g_nclusters 20 --g_online --g_min_size 100
```

test gluster
```
ipython -m unittest -- test_gluster.TestGlusterConv.test_mnist_online_delayed
```

## Dec 12
gluster batch
```
CUDA_VISIBLE_DEVICES=1 ipython --pdb -- bgluster.py --dataset imagenet --arch resnet34 --run_dir runs/resnet34_act --gb_citers 20 --pretrained --test_batch_size 64 --g_nclusters 20 --g_no_grad
CUDA_VISIBLE_DEVICES=2 ipython --pdb -- bgluster.py --dataset imagenet --arch resnet34 --run_dir runs/resnet34_fc --gb_citers 20 --pretrained --test_batch_size 256 --g_nclusters 20 --g_active_only 'model.module.fc'
CUDA_VISIBLE_DEVICES=2 ipython --pdb -- bgluster.py --dataset imagenet --arch resnet34 --run_dir runs/resnet34_input0 --gb_citers 20 --pretrained --test_batch_size 128 --g_nclusters 20 --g_no_grad --g_active_only 'model.module.conv1'
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- bgluster.py --dataset imagenet --arch resnet34 --run_dir runs/resnet34_seed2 --gb_citers 20 --pretrained --test_batch_size 8 --g_nclusters 20 --seed 2
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- bgluster.py --dataset imagenet --arch resnet34 --run_dir runs/resnet34 --gb_citers 20 --pretrained --test_batch_size 64 --g_nclusters 20
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- bgluster.py --dataset imagenet --arch vgg19 --run_dir runs/vgg19 --gb_citers 20 --pretrained --test_batch_size 8 --g_nclusters 20
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- bgluster.py --dataset imagenet --arch alexnet --run_dir runs/alexnet --gb_citers 20 --pretrained --test_batch_size 256 --g_nclusters 20
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- bgluster.py --dataset imagenet --arch resnet34 --run_dir runs/resnet34_online --gb_citers 20 --pretrained --test_batch_size 64 --g_nclusters 20 --g_online
```

mnist train
```
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main.py --dataset mnist --optim sgd --lr 0.1 --momentum 0.9   --logger_name runs/mnist --epochs 2
```

mnist online
```
CUDA_VISIBLE_DEVICES=1 ipython --pdb -- main.py --dataset mnist --optim sgd --lr 0.01 --momentum 0.9   --logger_name runs/Y --epochs 10 --gluster --gluster_num 3 --gluster_beta 0.9

## Jan 22

lr=[0.1,0.01,0.001], mom=[0.1, 0.5, 0.9], dmom=[0.1, 0.5, 0.9]
```
grid_run.py
```

lr=0.01, mom=0.5

```
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- main.py --logger_name runs/dmom_0.1/ --optim dmom --dmom 0.1
CUDA_VISIBLE_DEVICES=1 ipython --pdb -- main.py --logger_name runs/dmom_0.5/ --optim dmom --dmom 0.5
CUDA_VISIBLE_DEVICES=2 ipython --pdb -- main.py --logger_name runs/dmom_0.9/ --optim dmom --dmom 0.9
CUDA_VISIBLE_DEVICES=3 ipython --pdb -- main.py --logger_name runs/mom/ --optim sgd
```
