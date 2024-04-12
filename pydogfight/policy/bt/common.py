from __future__ import annotations

import json
import typing

from pydogfight.policy.bt.base_class import *
from pydogfight.core.actions import Actions
import pybts
from pybts import Status
import os
import numpy as np


def delay_updater(env: Dogfight2dEnv, time: float, status: Status) -> typing.Optional[Status]:
    """
    # 等待一段时间
    :param env: 环境
    :param time: 等待的时间
    :param status: 等待时间中的状态
    :return:
    """
    start_time = env.time
    while env.time - start_time < time:
        yield status


def go_to_location_updater(self: BTPolicyNode, location: tuple[float, float] | np.ndarray | list[float],
                           keep_time: int = 10) -> \
        typing.Optional[Status]:
    start_time = self.env.time
    self.actions.put_nowait((Actions.go_to_location, location[0], location[1]))
    yield Status.RUNNING
    while not self.agent.is_reach_location(location):
        yield Status.RUNNING
        if self.env.time - start_time >= keep_time:
            # 最多持续探索10秒
            self.put_update_message('最多持续运行10秒')
            break
        if not self.agent.is_current_route_target(location):
            # 当前目标点改变了
            self.put_update_message('当前目标点改变了')
            yield Status.FAILURE
            return
    yield Status.SUCCESS
    return
