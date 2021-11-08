import copy
import datetime
import importlib
import json
import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union

from run_modes import constants


def get_logger(experiment_path: str, name: str) -> logging.Logger:
    """Produce python logger.

    Args:
        experiment_path: path to save of log file.
        name: name of logger.

    Returns:
        logger: logging object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(constants.LOG_FORMAT)

    file_handler = logging.FileHandler(
        os.path.join(experiment_path, constants.LOG_FILE_NAME)
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def setup_experiment(
    mode: str,
    results_folder: str,
    config_path: str,
    config_changes_path: Optional[str] = None,
    seeds: Optional[List[int]] = None,
) -> Union[str, List[str]]:
    """Method to construct tree of paths for experiment outputs.

    Args:
        mode: name of experiment mode (single, serial, parallel etc.).
        results_folder: location of all results.
        config_path: path to original yaml configuration file.
        config_changes_path: path to file containing changes to be made to config.
        seeds: list of seeds over which to repeat experiment.

    Returns:
        paths: single path or list of paths for each sub-experiment.

    Raises:
        ValueError: if neither seeds nor config_changes_path are given
        and mode is not single.
        ValueError: if mode is not recognised.
    """
    timestamp = get_experiment_timestamp()
    experiment_path = os.path.join(results_folder, timestamp)

    os.makedirs(name=experiment_path, exist_ok=True)
    config_copy_path = os.path.join(experiment_path, "config.yaml")
    shutil.copyfile(config_path, config_copy_path)

    if mode == constants.SINGLE:
        paths = _setup_single_experiment(experiment_path=experiment_path)
    elif mode in [
        constants.MULTI,
        constants.SERIAL,
        constants.PARALLEL,
        constants.CLUSTER,
    ]:
        if config_changes_path is not None and seeds is not None:
            paths = _setup_multi_seed_changes_experiment(
                experiment_path=experiment_path,
                config_changes_path=config_changes_path,
                seeds=seeds,
            )
        elif config_changes_path is None and seeds is not None:
            paths = _setup_multi_seed_no_changes_experiment(
                experiment_path=experiment_path, seeds=seeds
            )
        elif config_changes_path is not None and seeds is None:
            paths = _setup_changes_experiment(
                experiment_path=experiment_path, config_changes_path=config_changes_path
            )
        else:
            raise ValueError(
                "For any run mode with multiple sub-runs, require either specification of config changes or seeds"
            )
    else:
        raise ValueError(f"run mode {mode} not recognised. Unable to setup_experiment.")
    return paths


def set_random_seeds(seed: int, packages: List[str]) -> None:
    """Set seeds for packages with non-deterministic behaviour.

    Each package has different method signature for setting seed, so each
    needs to be handled individually.
    Args:
        seed: seed to set.
        packages: list of packages to import and set seeds for.

    Raises:
        ValueError if package in list provided is not recognised.
    """
    managed_packages = []

    if constants.NUMPY in packages:
        import numpy as np

        np.random.seed(seed)
        managed_packages.append(constants.NUMPY)
    if constants.TORCH in packages:
        import torch

        torch.manual_seed(seed)
        managed_packages.append(constants.TORCH)
    if constants.RANDOM in packages:
        import random

        random.seed(seed)
        managed_packages.append(constants.RANDOM)

    unmanaged_packages = [p for p in packages if p not in managed_packages]
    if unmanaged_packages:
        raise ValueError(
            f"Packages put up for seed setting not covered: {unmanaged_packages}"
        )


def set_device(
    gpu_id: Union[int, None], logger: Optional[logging.Logger] = None
) -> Tuple[bool, Union[str, None]]:
    """Establish availability of specified devices.

    Behaviour is to default to CPU if GPUs cannot be found.

    Args:
        gpu_id: id of GPU to use. None if no GPU desired.
        logger: optionally provide a logger to output info.

    Returns:
        using_gpu: bool indicating whether the gpu is actually being used.
        experiment_device: specific device being used (None if torch unavailable).
    """
    if logger is not None:
        print_fn = logger.info
    else:
        print_fn = print
    try:
        import torch

        if gpu_id is not None:
            print_fn("Attempting to find GPU...")
            if torch.cuda.is_available():
                print_fn("GPU found, using the GPU...")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                using_gpu = True
                experiment_device = torch.device("cuda:{}".format(gpu_id))
                print_fn(f"Device in use: {experiment_device}")
            else:
                print_fn("GPU not found, reverting to CPU")
                using_gpu = False
                experiment_device = torch.device("cpu")
        else:
            print_fn("Using the CPU")
            experiment_device = torch.device("cpu")
            using_gpu = False
    except ModuleNotFoundError:
        print_fn("Torch not found, no changes made to devices.")
        experiment_device = None
        using_gpu = False
    return using_gpu, experiment_device


def config_changes_to_json(config_changes: List[Dict], json_path: str) -> None:
    """Write a list of dictionaries to json file.

    Args:
        config_changes: list of config change dictionaries.
        json_path: path of file to write to.
    """
    with open(json_path, "w") as json_file:
        json.dump(config_changes, json_file)


def get_experiment_timestamp() -> str:
    """Get a timestamp in YY-MM-DD-HH-MM-SS format."""
    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return exp_timestamp


def get_checkpoint_path(
    folder: str,
    timestamp: str,
    experiment_name: str,
    subfolder_name: Optional[str] = "",
) -> str:
    """Get full checkpoint path for experiment logs etc."""
    checkpoint_path = os.path.join(folder, timestamp, experiment_name, subfolder_name)
    return checkpoint_path


def _organise_config_changes_and_checkpoint_dirs(
    experiment_path: str,
    config_changes: Dict[str, List[Dict]],
    seeds: List[int],
) -> List[str]:
    """Method to organise paths for combination of different
    config changes and different seeds.

    Args:
        experiment_path: overall experiment path.
        config_changes: specification of config changes.
        seeds: list of seeds over which experiment is to be repeated.

    Returns:
        checkpoint_paths: list of output paths.
    """
    checkpoint_paths = []
    for i, (run_name, changes) in enumerate(config_changes.items()):
        for j, seed in enumerate(seeds):
            changes_copy = copy.deepcopy(changes)
            if seed is None:
                seed_path = constants.SINGLE
            else:
                seed_path = str(seed)
                changes_copy.append({constants.SEED: seed})
            checkpoint_path = os.path.join(experiment_path, run_name, seed_path)
            os.makedirs(name=checkpoint_path, exist_ok=True)
            config_changes_to_json(
                config_changes=changes_copy,
                json_path=os.path.join(checkpoint_path, constants.CONFIG_CHANGES_JSON),
            )
            checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


def _setup_single_experiment(experiment_path: str) -> str:
    """Generate single checkpoint path.

    Args:
        experiment_path: overall experiment path.

    Returns:
        single_checkpoint_path: path to individual experiment outputs.
    """
    single_checkpoint_path = os.path.join(experiment_path, constants.SINGLE)
    os.makedirs(name=single_checkpoint_path, exist_ok=True)
    return single_checkpoint_path


def _parse_config_changes(
    experiment_path: str, config_changes_path: str
) -> Dict[str, List]:
    """Method for extracting config changes from path, and saving
    them in overall experiment results folder.

    Args:
        experiment_path: overall experiment path.
        config_changes_path: path to file containing changes to be made to config.

    Returns:
        config_changes: mapping of sub-experiments to config adaptations.
    """
    config_changes = importlib.import_module(
        name=f"{config_changes_path.replace('.py', '')}"
    ).CONFIG_CHANGES
    config_changes_to_json(
        config_changes=config_changes,
        json_path=os.path.join(experiment_path, f"all_{constants.CONFIG_CHANGES_JSON}"),
    )
    return config_changes


def _setup_changes_experiment(
    experiment_path: str, config_changes_path: str
) -> List[str]:
    """Method for constructing paths in case of config changes
    but no seed variation.

    Args:
        experiment_path: overall experiment path.
        config_changes_path: path to file containing changes to be made to config.

    Returns:
        checkpoint_paths: list of checkpoint paths for different sub-experiments.
    """
    config_changes = _parse_config_changes(
        experiment_path=experiment_path, config_changes_path=config_changes_path
    )
    checkpoint_paths = _organise_config_changes_and_checkpoint_dirs(
        experiment_path=experiment_path, config_changes=config_changes, seeds=[None]
    )
    return checkpoint_paths


def _setup_multi_seed_changes_experiment(
    experiment_path: str, config_changes_path: str, seeds: List[int]
) -> List[str]:
    """Method for constructing paths in case of config changes and seed variation.

    Args:
        experiment_path: overall experiment path.
        config_changes_path: path to file containing changes to be made to config.
        seeds: list of seeds over which experiment is to be repeated.

    Returns:
        checkpoint_paths: list of checkpoint paths for different sub-experiments.
    """
    config_changes = _parse_config_changes(
        experiment_path=experiment_path, config_changes_path=config_changes_path
    )
    checkpoint_paths = _organise_config_changes_and_checkpoint_dirs(
        experiment_path=experiment_path,
        config_changes=config_changes,
        seeds=seeds,
    )
    return checkpoint_paths


def _setup_multi_seed_no_changes_experiment(
    experiment_path: str, seeds: List[int]
) -> List[str]:
    """Method for constructing paths in case of seed variation but no config changes.

    Args:
        experiment_path: overall experiment path.
        seeds: list of seeds over which experiment is to be repeated.

    Returns:
        checkpoint_paths: list of checkpoint paths for different sub-experiments.
    """
    checkpoint_paths = []
    for seed in seeds:
        path = os.path.join(experiment_path, str(seed))
        checkpoint_paths.append(path)
        os.makedirs(path, exist_ok=True)
        # placeholder, empty file
        config_changes_to_json(
            config_changes={},
            json_path=os.path.join(path, constants.CONFIG_CHANGES_JSON),
        )
    return checkpoint_paths


def json_to_config_changes(json_path: str) -> List[Dict]:
    """Read a list of dictionaries from json file.

    Args:
        json_path: path to json file

    Returns:
        config_changes: list of config change mappings.
    """
    with open(json_path, "r") as json_file:
        config_changes = json.load(json_file)
    return config_changes


def process_seed_arguments(seeds: Union[str, List[int]]):
    """Seed specification from command line often in string format.

    Args:
        seeds: str encoding of seeds.

    Returns:
        seeds: list of seed ints.
    """
    if isinstance(seeds, list):
        return seeds
    elif isinstance(seeds, str):
        try:
            seeds = [int(seeds.strip("[").strip("]"))]
        except ValueError:
            seeds = [int(s) for s in seeds.strip("[").strip("]").split(",")]
    return seeds


def create_job_script(
    run_command: str,
    save_path: str,
    num_cpus: int,
    conda_env_name: str,
    memory: int,
    num_gpus: int,
    gpu_type: str,
    error_path: str,
    output_path: str,
    array_job_length: int = 0,
    walltime: str = "24:0:0",
) -> None:
    """Create a job script for use on HPC using Univa.

    Args:
            run_command: main script command, e.g. 'python run.py'
            save_path: path to save the job script to
            num_cpus: number of cores for job
            conda_env_name: name of conda environment to activate for job
            memory: number of gb memory to allocate to node.
            walltime: time to give job--1 day by default
    """
    with open(save_path, "w") as file:
        resource_specification = f"#PBS -lselect=1:ncpus={num_cpus}:mem={memory}gb"
        if num_gpus:
            resource_specification += f":ngpus={num_gpus}:gpu_type={gpu_type}"
        file.write(f"{resource_specification}\n")
        file.write(f"#PBS -lwalltime={walltime}\n")
        if array_job_length:
            file.write(f"#PBS -J 1-{array_job_length}\n")
        # output/error file paths
        file.write(f"#PBS -e {error_path}\n")
        file.write(f"#PBS -o {output_path}\n")
        # initialise conda env
        file.write("module load anaconda3/personal\n")
        file.write(f"source activate {conda_env_name}\n")
        # change to dir where job was submitted from
        file.write("cd $PBS_O_WORKDIR\n")
        # job script
        file.write(f"{run_command}\n")


def create_slurm_job_script(
    run_command: str,
    save_path: str,
    num_cpus: int,
    conda_env_name: str,
    memory: int,
    num_gpus: int,
    gpu_type: str,
    error_path: str,
    output_path: str,
    array_job_length: int = 0,
    walltime: str = "24:0:0",
) -> None:
    """Create a job script for use on HPC using SLURM.

    Args:
            run_command: main script command, e.g. 'python run.py'
            save_path: path to save the job script to
            num_cpus: number of cores for job
            conda_env_name: name of conda environment to activate for job
            memory: number of gb memory to allocate to node.
            walltime: time to give job--1 day by default
    """
    with open(save_path, "w") as file:
        file.write("#!/bin/bash\n")
        # num CPUs
        file.write(f"#SBATCH --ntasks={num_cpus}\n")
        # memory (GB)
        file.write(f"#SBATCH --mem={memory}gb\n")
        # walltime
        file.write(f"#SBATCH --time={walltime}\n")
        # out file
        file.write(f"#SBATCH --output={output_path}\n")
        # err file
        file.write(f"#SBATCH --error={error_path}\n")

        # command
        file.write(f"{run_command}\n")
