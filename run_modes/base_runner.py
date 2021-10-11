import abc
from typing import List

from data_logger import data_logger
from run_modes import utils


class BaseRunner(abc.ABC):
    """Abstract base class for a runner that can be used with
    the various run modes.

    Abstract methods:
        - _get_data_columns
    """

    def __init__(self, config) -> None:
        """Class constructor.
        Creates data logger instance with columns obtained from
        output of method implemented in child class.
        Creates logger instance for standard logging.

        Args:
            config: configuration object.
        """
        self._checkpoint_path = config.checkpoint_path
        self._data_logger = data_logger.DataLogger(
            checkpoint_path=self._checkpoint_path,
            logfile_path=config.logfile_path,
            columns=self._get_data_columns(),
        )
        self._logger = utils.get_logger(
            experiment_path=self._checkpoint_path, name=__name__
        )

    @abc.abstractmethod
    def _get_data_columns(self) -> List[str]:
        """Output data columns to be logged by runner."""
        pass
