import abc
from typing import List, Type

from config_manager import base_configuration
from data_logger import data_logger
from plotter import plotter
from run_modes import utils


class BaseRunner(abc.ABC):
    """Abstract base class for a runner that can be used with
    the various run modes.

    Abstract methods:
        - _get_data_columns
    """

    def __init__(
        self, config: Type[base_configuration.BaseConfiguration], unique_id: str = ""
    ) -> None:
        """Class constructor.
        Creates data logger instance with columns obtained from
        output of method implemented in child class.
        Creates logger instance for standard logging.

        Args:
            config: configuration object.
        """
        if unique_id != "":
            name = f"{__name__}.{unique_id}"
            logfile_path_name = config.logfile_path.split(".csv")[0]
            logfile_path = f"{logfile_path_name}_{unique_id}.csv"
        else:
            name = __name__
            logfile_path = config.logfile_path

        self._checkpoint_path = config.checkpoint_path

        self._data_logger = data_logger.DataLogger(
            checkpoint_path=self._checkpoint_path,
            logfile_path=logfile_path,
            columns=self._get_data_columns(),
        )

        self._logger = utils.get_logger(
            experiment_path=self._checkpoint_path, name=name
        )
        self._plotter = plotter.Plotter(
            save_folder=self._checkpoint_path,
            logfile_path=config.logfile_path,
            smoothing=config.smoothing,
            xlabel=config.xlabel,
        )

    @abc.abstractmethod
    def _get_data_columns(self) -> List[str]:
        """Output data columns to be logged by runner."""
        pass
