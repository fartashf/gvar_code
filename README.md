# A Study of Gradient Variance in Deep Learning (original raw repository)
(Here is a lazy way of open-sourcing all the code for a multi-year long 
project.)
Code in this repository contains the implementation of a collection of ideas on 
improving and analysis of optimization methods. That includes the experiments 
for the following paper:
**[A Study of Gradient Variance in Deep 
Learning](https://arxiv.org/abs/2007.04532)**
*, F. Faghri, D. Duvenaud, D. J. Fleet, J. Ba, arXiv:2007.04532*.

## Features
The following ideas might be interesting to different researchers.


### All experiments and notes are in Jupyter notebooks

See `notebooks/figures*.ipynb` for all the records of what was tried, what 
failed, what worked, and all the figures. These are probably the main reason 
I am open-sourcing this code the lazy way so that all of it is there if someone 
  wants to to inspect further.


### Grid run

See [`grid_run`](https://github.com/fartashf/dmom_code/blob/master/grid_run.py) 
and 
[`cluster.py`](https://github.com/fartashf/dmom_code/blob/master/grid/cluster.py) 
for a light-weight grid-search code. An example of using the code is 
[here](https://github.com/fartashf/dmom_code/blob/master/grid/icml.py).

### Abstraction of Iterative Optimization Methods

(For a cleaner implementation of this abstraction see 
[FOptim](https://github.com/fartashf/foptim) repository.)

In order to accurately measure statistics of gradients, we need to make sure 
the logging does not interfere with internal operations of an optimizer. For 
example, the sampling of the data should not be affected by the sampling for 
estimating statistics. Also some optimizers such as K-FAC have periodic 
operations that should not be performed while measuring statistics. Our 
abstraction facilitates these.

Here we describe an abstraction of the following implemented optimizers: SGD, 
SGD+Momentum, Adam, K-FAC, and the variance reduction methods SVRG and our 
proposed GC sampler/optimizer. An optimization method has a major operation 
`step` [step 
function](https://github.com/fartashf/dmom_code/blob/96a245247d111b2fa40a1efe3d988cc1a4aadf72/estim/optim.py#L81) 
that is repeatedly executed that relies on a direction and a step size/learning 
rate.  It can also include some 
[frequent](https://github.com/fartashf/dmom_code/blob/96a245247d111b2fa40a1efe3d988cc1a4aadf72/estim/optim.py#L105) 
or [infrequent 
operations](https://github.com/fartashf/dmom_code/blob/96a245247d111b2fa40a1efe3d988cc1a4aadf72/estim/optim.py#L110).  
Frequent updates are as cheap as a single gradient calculation and their cost 
becomes negligible if done every 10-100 iterations. Infrequent "snapshots" have 
a cost relative to a full pass over the training set and can be amortized out 
if done only a few times during the entire training.
K-FAC is an example of an optimizer that has both types of updates. See NTK 
branch for that implementation 
[code](https://github.com/fartashf/dmom_code/blob/b840e0ac8eb0182de58331e46bec46406d32fb4c/estim/kfac.py#L69).

The important part of this abstraction is that each optimizer has to have 
a `grad()` function that returns the step direction specific to the optimizer.  
For example, in K-FAC, the proposed direction is the preconditioned gradient 
[code](https://github.com/fartashf/dmom_code/blob/b840e0ac8eb0182de58331e46bec46406d32fb4c/estim/kfac.py#L23).  
In SVRG it is the gradient after adding the control variate and subtracting the 
old values of the gradient of the current mini-batch 
[code](https://github.com/fartashf/dmom_code/blob/96a245247d111b2fa40a1efe3d988cc1a4aadf72/estim/svrg.py#L50).

This abstraction allows us to measure statistics of the step direction that is 
studied in the above paper. For that, we have the 
[`grad_estim`](https://github.com/fartashf/dmom_code/blob/b840e0ac8eb0182de58331e46bec46406d32fb4c/estim/gestim.py#L38) 
function that allows us to evaluate gradient multiple times during training 
while not affecting the internal operations of an optimizer. The statistics are 
measured by calls to `grad_estim` in 
[`get_Ege_var`](https://github.com/fartashf/dmom_code/blob/b840e0ac8eb0182de58331e46bec46406d32fb4c/estim/gestim.py#L49) 
function that is called by 
[`log_var`](https://github.com/fartashf/dmom_code/blob/b840e0ac8eb0182de58331e46bec46406d32fb4c/estim/gvar.py#L164) 
function that is called from the main training loop along with other logging 
calls 
[code](https://github.com/fartashf/dmom_code/blob/96a245247d111b2fa40a1efe3d988cc1a4aadf72/main/gvar.py#L85).


### Gradient Clustering (GC or Gluster in this repo)

Gradient Clustering is an efficient method for clustering gradients during 
training. Using stratified sampling we achieve low variance unbiased gradient 
estimators. The non-weighted implementation is also a tool for inspecting 
gradients of a model and understanding its decisions.

The main classes are in [gluster 
directory](https://github.com/fartashf/dmom_code/tree/master/gluster) and 
[gluster 
estimator](https://github.com/fartashf/dmom_code/blob/master/estim/gluster.py).  
We have implemented both an online and a full batch version of this gradient 
estimator. Stratified sampling is implemented 
[here](https://github.com/fartashf/dmom_code/blob/96a245247d111b2fa40a1efe3d988cc1a4aadf72/data.py#L643) 
and relies on a slightly modified dataset wrapper that returns indices 
[code](https://github.com/fartashf/dmom_code/blob/96a245247d111b2fa40a1efe3d988cc1a4aadf72/data.py#L106).  
This requirement for indices is why we have some challenges in applying this 
idea to infinite data streams and large datasets but there are solutions that 
we have implemented separately and will include in another branch.


### Zero gradients and Ad-hoc sampling

We implemented a lot of ad-hoc ideas for important sampling of data points 
according to their loss value or norm of the gradient. All these ideas fail on 
large datasets but they all work on MNIST. For those ideas see 
[here](https://github.com/fartashf/dmom_code/blob/master/schedulers.py),
[here](https://github.com/fartashf/dmom_code/blob/master/data.py) and 
[here](https://github.com/fartashf/dmom_code/blob/master/optim/dmom.py).

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 2.7
* [PyTorch](http://pytorch.org/) (>1.4.0)
* [NumPy](http://www.numpy.org/) (>1.15.4)
* [torchvision]()
* [matplotlib]()


## Reference

If you found this code useful, please cite the following paper:

    @misc{faghri2020study,
        title={A Study of Gradient Variance in Deep Learning},
        author={Fartash Faghri and David Duvenaud and David J. Fleet and Jimmy Ba},
        year={2020},
        eprint={2007.04532},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
