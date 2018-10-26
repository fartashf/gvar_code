# Data Momentum or ?
This repository contains optimization methods for dealing with two simple cases 
of training data:

* Zero gradients
* Near duplicates gradients

## Zero Gradients
The underlying observation is we can remove training data with zero gradients, 
at least for one iteration, and achieve an unbiased gradient estimator with 
lower variance.

Methods based on this observation are the following:

* Data Momentum (`optim/{dmom,add_dmom,jvp}.py`)
* Snoozer and samplers (`schedulers.py`)

## Near duplicate gradients
We observe that if we have duplicate data points, we can replace all 
occurrences with one instance and get an unbiased gradient estimator with the 
same variance by multiplying the gradient of that instance with the number of 
duplicates.

* Gluster (`gluster.py`)

## Start From

* `notebooks/figures*.ipynb`: Analysis and visualization
* `main.py`: Running a single experiment
* `grid_run.py`: Grid search and the configurations of grid searches so far.
* `test/`: some unit tests.
