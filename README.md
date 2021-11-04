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

A class of type BaseRunner and a configuration object that extends the BaseConfiguration class provided in [this configuration package](https://github.com/seblee97/config_package), are the primary arguments that need to be provided to one of the numerous "run mode" classes in this package.