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
