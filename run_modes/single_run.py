"""
This script is for running a single experiment.

It can either be called from the command line with the following required arguments:

    - path pointing to the configuration yaml file
    - path pointing to the firectory in which to output results of experiment

Optionally, the following arguments can be provided:

    - path pointing to a json file describing changes to be made  to the config.

Else, the individual method 'single_run' can be imported for use in other workflows,
e.g. to use the multiprocessing module.
"""
import os
import re
from typing import Dict, List, Type

from config_manager import base_configuration
from run_modes import base_runner, constants, utils


def single_run(
    runner_class: Type[base_runner.BaseRunner],
    config_class: Type[base_configuration.BaseConfiguration],
    run_methods: List[str],
    config_path: str,
    checkpoint_path: str,
    changes: List[Dict] = [],
    stochastic_packages: List[str] = [],
) -> None:
    """Single experiment run.

    Args:
        runner_class: runner class to be instantiated.
        config_class: configuration class to be instantiated.
        run_methods: list of methods (by name) belonging to the runner
        that are to be called.
        config_path: path to yaml configuration file for experiment.
        checkpoint_path: path to directories to output results.
        changes: changes to be made to config.
        stochastic_packages: list of packages (by name) for which seeds are to be set.
    """
    # instantiate logging module.
    # use unique id here to ensure separate loggers for each runner.
    unique_id = re.sub(r"\W+", "", str(changes))

    logger = utils.get_logger(
        experiment_path=checkpoint_path, name=f"{__name__}.{unique_id}"
    )

    config = config_class(config=config_path, changes=changes)

    # default runner config values
    try:
        seed = getattr(config, constants.SEED)
        config.amend_property(property_name=constants.SEED, new_property_value=seed)
    except AttributeError:
        seed = 0
        config.add_property(property_name=constants.SEED, property_value=seed)
    try:
        gpu_id = getattr(config, constants.GPU_ID)
    except AttributeError:
        gpu_id = None
    try:
        xlabel = getattr(config, constants.XLABEL)
    except AttributeError:
        xlabel = "X"
        config.add_property(property_name=constants.XLABEL, property_value=xlabel)
    try:
        smoothing = getattr(config, constants.SMOOTHING)
    except AttributeError:
        smoothing = 1
        config.add_property(property_name=constants.SMOOTHING, property_value=smoothing)

    # configure random seeds
    utils.set_random_seeds(seed=seed, packages=stochastic_packages)

    # configure device (cpu vs. gpu etc.)
    using_gpu, experiment_device = utils.set_device(gpu_id=gpu_id, logger=logger)
    config.add_property(constants.USING_GPU, using_gpu)
    config.add_property(constants.EXPERIMENT_DEVICE, experiment_device)

    config.add_property(constants.CHECKPOINT_PATH, checkpoint_path)
    config.add_property(
        constants.LOGFILE_PATH,
        os.path.join(checkpoint_path, "data_logger.csv"),
    )

    config.save_configuration(folder_path=checkpoint_path)

    runner = runner_class(config=config, unique_id=unique_id)

    for run_method in run_methods:
        try:
            method = getattr(runner, run_method)
        except AttributeError:
            print(f"Method with name {run_method} not found on object {runner}")
        method()
