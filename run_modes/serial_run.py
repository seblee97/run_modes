import os
from typing import List, Type

from config_manager import base_configuration
from run_modes import base_runner, constants, single_run, utils


def serial_run(
    runner_class: Type[base_runner.BaseRunner],
    config_class: Type[base_configuration.BaseConfiguration],
    run_methods: List[str],
    config_path: str,
    checkpoint_paths: List[str],
    stochastic_packages: List[str] = [],
):
    """Set of experiments run in serial using multiprocessing module.

    Args:
        runner_class: runner class to be instantiated.
        config_class: configuration class to be instantiated.
        run_methods: list of methods to be called on runner class.
        config_path: path to yaml configuration file for experiment.
        checkpoint_paths: list of paths to directories to output results.
        stochastic_packages: list of packages (by name) for which seeds are to be set.
    """
    for checkpoint_path in checkpoint_paths:
        changes = utils.json_to_config_changes(
            os.path.join(checkpoint_path, constants.CONFIG_CHANGES_JSON)
        )
        single_run.single_run(
            runner_class=runner_class,
            config_class=config_class,
            run_methods=run_methods,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            changes=changes,
            stochastic_packages=stochastic_packages,
        )
