"""
This script is for runs being executed on the cluster.

While the cluter_run.py deals with submitting one job and parallelising within that job,
this script is for submitting a range of jobs.
"""
import os
import subprocess
from typing import List

from run_modes import constants, utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def cluster_run(
    runner_class_name: str,
    runner_module_name: str,
    runner_module_path: str,
    config_class_name: str,
    config_module_name: str,
    config_module_path: str,
    run_methods: List[str],
    config_path: str,
    checkpoint_paths: List[str],
    scheduler: str = "slurm",
    num_cpus: int = 4,
    num_gpus: int = 0,
    memory: int = 16,
    walltime: str = "24:0:0",
    gpu_type: str = "K80",
    env_name: str = "",
    stochastic_packages: List[str] = [],
    cluster_debug: bool = False,
) -> None:
    """Set of experiments run in parallel on a cluster.

    Args:
        runner_class_name: name of runner class to be imported and
        ultimately instantiated.
        runner_module_name: name of module in which runner class is defined.
        runner_module_path: path pointing to runner module.
        config_class_name: name of config class to be imported and
        ultimately instantiated.
        config_module_name: name of module in which config class is defined.
        config_module_path: path pointing to config module.
        run_methods: list of methods to be called on runner class.
        config_path: path to yaml configuration file for experiment.
        checkpoint_paths: list of paths to directories to output results.
        num_cpus: number of CPUs to request per job.
        num_gpus: number of GPUs to request per job.
        memory: memory per node (in GB) to request for each job.
        gpu_type: type of GPU type to request.
        env_name: name of conda environment to activate.
        stochastic_packages: list of packages (by name) for which seeds are to be set.
        cluster_debug: bool to indicate whether to test pipeline locally.
    """
    for checkpoint_path in checkpoint_paths:
        changes_path = os.path.join(checkpoint_path, constants.CONFIG_CHANGES_JSON)
        job_script_path = os.path.join(checkpoint_path, constants.JOB_SCRIPT)

        error_path = os.path.join(checkpoint_path, constants.ERROR_FILE_NAME)
        output_path = os.path.join(checkpoint_path, constants.OUTPUT_FILE_NAME)

        run_script_path = os.path.join(MAIN_FILE_PATH, "command_line_run.py")

        run_command = (
            f"python {run_script_path} --config_path {config_path} "
            f"--checkpoint_path {checkpoint_path} "
            f"--run_methods '{run_methods}' "
            f"--runner_class_name {runner_class_name} "
            f"--runner_module_name {runner_module_name} "
            f"--runner_module_path {runner_module_path} "
            f"--config_class_name {config_class_name} "
            f"--config_module_name {config_module_name} "
            f"--config_module_path {config_module_path} "
            f"--config_changes_path {changes_path} "
            f"--stochastic_packages '{stochastic_packages}'"
        )

        if scheduler == constants.SLURM:
            script_command = utils.create_slurm_job_script
            subprocess_call = "sbatch"
        elif scheduler == constants.UNIVA:
            script_command = utils.create_job_script
            subprocess_call = "qsub"

        script_command(
            run_command=run_command,
            save_path=job_script_path,
            num_cpus=num_cpus,
            conda_env_name=env_name,
            memory=memory,
            num_gpus=num_gpus,
            gpu_type=gpu_type,
            error_path=error_path,
            output_path=output_path,
            walltime=walltime,
        )

        if cluster_debug:
            subprocess.call(run_command, shell=True)
        else:
            subprocess.call(f"{subprocess_call} {job_script_path}", shell=True)
