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
