# Run Modes

This package provides a skeleton for machine learning experiments. The aim is to take care of some of the boiler plate code written in many machine learning experiment pipelines (configurations, runners, etc.).

## Getting Started

Installation can either be performed by cloning the repository and running ```pip install -e .``` from the package root, or via ```pip install -e git://github.com/seblee97/run_modes.git#egg=run-modes-seblee97```.

## Overview

Most machine learning experiment pipelines follow a similar structure: a runner class is constructed in which the dataset, loss function, optimiser etc. is setup and then train/test loops are called. In the meanwhile data is logged either to a csv-like file or tensorboard-type programme.
The idea of this package is to provide the backbone of this pipeline, and make it easy to scale up experimentation including on remote computing resources.

### BaseRunner

The first aspect of the package is the BaseRunner class in ```base_runner.py```. A runner that extends this base class gets access to a logger, data logger (to csv, provided by [this package](https://github.com/seblee97/data_logger)), and plotter (provided by [this package](https://github.com/seblee97/plotter)).

### Run Modes

A class of type BaseRunner and a configuration object that extends the BaseConfiguration class provided in [this configuration package](https://github.com/seblee97/config_package), are the primary arguments that need to be given to one of the numerous "run mode" classes in this package.

The simplest "run mode" is the single_run method, which performs a single experiment. Assuming a runner class called ExampleRunner and a config class called ExampleConfig, the usage for a pipeline with a _train_ method followed by a _test_ method would be as follows:

```
from run_modes import single_run

single_run.single_run(
		runner_class=ExampleRunner,
		config_class=ExampleConfig,
		config_path=path_to_yaml,
		checkpoint_path=path_to_exp,
		run_methods=["train", "test"],
	)
```

A more comprehensive example for a runner plotting simple polynomials can be found in the ```example``` folder. To see basic behaviour of the runner you can run ```python example/main.y```.

Other "run modes" provided are extensions to this single run that allow a range of experiments (in series or parallel), and submission to job schedulers such as SLURM or UNIVA.