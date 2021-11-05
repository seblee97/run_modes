import random
from typing import Type

from config_manager import base_configuration
from run_modes import base_runner


class ExampleRunner(base_runner.BaseRunner):
    def __init__(
        self, config: Type[base_configuration.BaseConfiguration], unique_id: str
    ):
        super().__init__(config=config, unique_id=unique_id)

        self._num_iterations = config.num_iterations

    def _get_data_columns(self):
        return ["linear", "quadratic"]

    def train(self):
        for i in range(self._num_iterations):
            y1 = 5 * i + random.random()
            y2 = i ** 2 + 3 * i + random.random()

            self._data_logger.write_scalar(tag="linear", step=i, scalar=y1)
            self._data_logger.write_scalar(tag="quadratic", step=i, scalar=y2)

            self._data_logger.checkpoint()

    def plot(self):
        self._plotter.load_data()
        self._plotter.plot_learning_curves()
