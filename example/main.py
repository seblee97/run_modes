import os

import example_config
import example_runner
from run_modes import single_run, utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":

    runner = example_runner.ExampleRunner
    config = example_config.ExampleConfig

    single_checkpoint_path = utils.setup_experiment(
        mode="single",
        results_folder=os.path.join(MAIN_FILE_PATH, "example_results"),
        config_path=os.path.join(MAIN_FILE_PATH, "config.yaml"),
    )

    single_run.single_run(
        runner_class=runner,
        config_class=config,
        run_methods=["train", "plot"],
        config_path=os.path.join(MAIN_FILE_PATH, "config.yaml"),
        checkpoint_path=single_checkpoint_path,
    )
