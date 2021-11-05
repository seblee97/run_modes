import example_config
import example_runner
from run_modes import single_run, utils

if __name__ == "__main__":

    runner = example_runner.ExampleRunner
    config = example_config.ExampleConfig

    single_checkpoint_path = utils.setup_experiment(
        mode="single", results_folder="example_results", config_path="config.yaml"
    )

    single_run.single_run(
        runner_class=runner,
        config_class=config,
        run_methods=["train", "plot"],
        config_path="config.yaml",
        checkpoint_path=single_checkpoint_path,
    )
