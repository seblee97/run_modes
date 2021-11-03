import argparse
import importlib.util
from typing import Any, Type

from config_manager import base_configuration
from run_modes import base_runner, single_run, utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config_path",
    metavar="-C",
    help="path to base configuration file.",
)
parser.add_argument(
    "--checkpoint_path", metavar="-CP", help="path to checkpointing for this run."
)
parser.add_argument(
    "--run_methods",
    metavar="-RM",
    help="list of methods to call on runner.",
)
parser.add_argument(
    "--stochastic_packages",
    metavar="-SP",
    help="list of packages with stochastic behvaiour to set seeds for.",
)
parser.add_argument("--runner_class_name", metavar="-RCN", help="Name of runner class.")
parser.add_argument(
    "--runner_module_name",
    metavar="-RMN",
    help="Name of module in which runner class is defined.",
)
parser.add_argument(
    "--runner_module_path", metavar="-RMP", help="Full path to runner module."
)
parser.add_argument("--config_class_name", metavar="-CCN", help="Name of config class.")
parser.add_argument(
    "--config_module_name",
    metavar="-CMN",
    help="Name of module in which config class is defined.",
)
parser.add_argument(
    "--config_module_path", metavar="-CMP", help="Full path to config module."
)
parser.add_argument(
    "--config_changes_path", metavar="-CC", help="Path to config changes."
)


def _import_exernal_module(class_name: str, module_spec) -> Any:
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    imported_class = getattr(module, class_name)

    return imported_class


def get_runner_class(
    runner_class_name: str, runner_spec
) -> Type[base_runner.BaseRunner]:
    runner_class = _import_exernal_module(
        class_name=runner_class_name, module_spec=runner_spec
    )
    return runner_class


def get_config_class(
    config_class_name: str, config_spec
) -> Type[base_configuration.BaseConfiguration]:
    config_class = _import_exernal_module(
        class_name=config_class_name, module_spec=config_spec
    )
    return config_class


if __name__ == "__main__":
    args = parser.parse_args()

    runner_spec = importlib.util.spec_from_file_location(
        args.runner_module_name, args.runner_module_path
    )
    config_spec = importlib.util.spec_from_file_location(
        args.config_module_name, args.config_module_path
    )

    runner_class = get_runner_class(
        runner_class_name=args.runner_class_name, runner_spec=runner_spec
    )
    config_class = get_config_class(
        config_class_name=args.config_class_name, config_spec=config_spec
    )

    run_methods = [
        method.strip() for method in args.run_methods.strip("[").strip("]").split(",")
    ]
    config_changes = utils.json_to_config_changes(args.config_changes_path)
    stochastic_packages = [
        package.strip()
        for package in args.stochastic_packages.strip("[").strip("]").split(",")
    ]

    single_run.single_run(
        runner_class=runner_class,
        config_class=config_class,
        run_methods=run_methods,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        changes=config_changes,
        stochastic_packages=stochastic_packages,
    )
