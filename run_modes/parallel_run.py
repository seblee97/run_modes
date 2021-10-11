import os
from typing import List

from run_modes import constants, single_run, utils

try:
    import torch.multiprocessing as mp
except ModuleNotFoundError:
    import multiprocessing as mp


def parallel_run(
    runner_class,
    config_class,
    run_methods: List[str],
    config_path: str,
    checkpoint_paths: List[str],
    stochastic_packages: List[str] = [],
) -> None:
    """Set of experiments run in parallel using multiprocessing module.

    Args:
        runner_class: runner class to be instantiated.
        config_class: configuration class to be instantiated.
        config_path: path to yaml configuration file for experiment.
        checkpoint_paths: list of paths to directories to output results.
        stochastic_packages: list of packages (by name) for which seeds are to be set.
    """
    processes = []

    for checkpoint_path in checkpoint_paths:
        changes = utils.json_to_config_changes(
            os.path.join(checkpoint_path, constants.CONFIG_CHANGES_JSON)
        )
        process = mp.Process(
            target=single_run.single_run,
            args=(
                runner_class,
                config_class,
                run_methods,
                config_path,
                checkpoint_path,
                changes,
                stochastic_packages,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
