from stable_baselines3.common.logger import configure, Logger
from typing import Any
from pydogfight.envs import Dogfight2dEnv
from pydogfight.core.world_obj import *


class CustomLogger:

    def __init__(self, filepath: str, verbose: int):
        self.logger = configure(filepath,
                                format_strings=["stdout", "tensorboard"] if verbose == 1 else ['tensorboard'])
        self.filepath = filepath
        self.verbose = verbose
        self.old_name_to_value = { }

    def record(self, key: str, value: Any) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        """
        if value is None:
            return
        self.logger.record(key, value)

    def record_minus_old(self, key: str, value: Any) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        """
        if value is None:
            return
        old_value = self.old_name_to_value.get(key, 0)
        self.logger.record(key, value - old_value)

    def record_sum_old(self, key: str, value: Any) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        """
        if value is None:
            return
        old_value = self.old_name_to_value.get(key, 0)
        self.logger.record(key, value + old_value)

    def dump(self, step: int) -> None:
        self.old_name_to_value = self.logger.name_to_value.copy()
        self.logger.dump(step=step)
