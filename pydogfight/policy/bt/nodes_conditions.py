from __future__ import annotations

import json
import typing

from pydogfight.core.models import BoundingBox

from pydogfight.policy.bt.nodes import *
from pydogfight.core.world_obj import Aircraft, Missile
from pydogfight.algos.traj import calc_optimal_path
from pydogfight.core.actions import Actions
import pybts
from pybts import Status
import os
from pydogfight.policy.bt.common import *


class IsWin(BTPolicyNode, pybts.Condition):

    def update(self) -> Status:
        info = self.env.gen_info()
        if info['winner'] == self.agent.color:
            return Status.SUCCESS
        return Status.FAILURE


class IsLose(BTPolicyNode, pybts.Condition):

    def update(self) -> Status:
        info = self.env.gen_info()
        if info['winner'] == self.agent.enemy_color:
            return Status.SUCCESS
        return Status.FAILURE


class IsDraw(BTPolicyNode, pybts.Condition):

    def update(self) -> Status:
        info = self.env.gen_info()
        if info['winner'] == 'draw':
            return Status.SUCCESS
        return Status.FAILURE


class IsReachLocation(BTPolicyNode, pybts.Condition):
    """由于策略的更新时间较长，可能无法正确判断是否到达某个目标点"""

    def __init__(self, x: float, y: float, **kwargs):
        super().__init__(**kwargs)
        self.x = self.converter.float(x)
        self.y = self.converter.float(y)

    def to_data(self):
        return {
            **super().to_data(),
            'x': self.x,
            'y': self.y,
        }

    def update(self) -> Status:
        reach_distance = self.agent.speed * max(
                self.env.options.delta_time,
                self.env.options.policy_interval * 2) * self.env.options.reach_location_threshold
        if self.agent.distance((self.x, self.y)) <= reach_distance:
            return Status.SUCCESS
        return Status.FAILURE


class IsNearEnemy(BTPolicyNode):
    """
    是否靠近敌机一定距离以内

    distance: 距离，单位m
    """

    def __init__(self, distance: float | str = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.distance = self.converter.float(self.distance)  # 以自己的雷达半径为判断依据

    def update(self) -> Status:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            return Status.FAILURE

        distance = self.agent.distance(enemy)

        if distance <= self.distance:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsInSafeArea(BTPolicyNode, pybts.Condition):
    """
    条件节点: 是否处于安全区域，如果不在安全区域，则要往战场中心飞
    - SUCCESS: 处于安全区域
    - FAILURE: 不处于
    """

    def to_data(self):
        return {
            **super().to_data(),
            'distance'            : self.agent.distance((0, 0)),
            'bullseye_safe_radius': self.env.options.bullseye_safe_radius()
        }

    def update(self) -> Status:
        if self.agent.distance((0, 0)) >= self.env.options.bullseye_safe_radius():
            return Status.FAILURE
        return Status.SUCCESS


class IsInGameRange(BTPolicyNode, pybts.Condition):
    """
    条件节点：是否在战场中
    """

    def update(self) -> Status:
        if self.agent.is_in_game_range:
            return Status.SUCCESS
        return Status.FAILURE


class IsOutOfGameRange(BTPolicyNode, pybts.Condition):
    """
    条件节点：是否不在战场中
    """

    def update(self) -> Status:
        if not self.agent.is_in_game_range:
            return Status.SUCCESS
        return Status.FAILURE


class IsOnActiveRoute(BTPolicyNode, pybts.Condition):
    """
    条件节点: 当前agent是否存在未完成的飞行路线
    - SUCCESS: 存在未完成的飞行路线
    - FAILURE: 不存在
    """

    def update(self) -> Status:
        if self.agent.route is not None:
            return Status.SUCCESS
        return Status.FAILURE


class IsEnemyDetected(BTPolicyNode, pybts.Condition):
    """
    行为节点：是否监测到敌机
    - SUCCESS: 监测到敌机
    - FAILURE: 没有监测到敌机
    """

    def update(self) -> Status:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            return Status.FAILURE

        return Status.SUCCESS


class IsMissileThreatDetected(BTPolicyNode, pybts.Condition):
    """
    条件节点:是否发现来袭导弹，如果发现了，可能需要进行规避
    - SUCCESS: 发现来袭导弹
    - FAILURE: 未发现
    """

    def update(self) -> Status:
        missiles = self.env.battle_area.detect_missiles(agent_name=self.agent_name)
        if len(missiles) > 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsMissileDepleted(BTPolicyNode, pybts.Condition):
    """
    条件节点: 导弹是否用完了，用完了就要返回基地补充导弹
    - SUCCESS: 剩余导弹 == 0
    - FAILURE: 剩余导弹 > 0
    """

    def update(self) -> Status:
        if self.agent.missile_count == 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsMissileFull(BTPolicyNode, pybts.Condition):
    """
    条件节点: 剩余导弹是否是满的
    - SUCCESS: 剩余导弹 == 0
    - FAILURE: 剩余导弹 > 0
    """

    def update(self) -> Status:
        if self.agent.missile_count == self.agent.options.aircraft_missile_count:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsFuelDepleted(BTPolicyNode, pybts.Condition):
    """
    条件节点：检查燃油是否用完
    - SUCCESS: 剩余油量 <= 0
    - FAILURE: 否则
    """

    def update(self) -> Status:
        if self.agent.fuel <= 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsFuelBingo(BTPolicyNode, pybts.Condition):
    """
    条件节点：检查是否达到了bingo油量（紧急燃油水平）。如果处于bingo油量以下，则需要返航
    - SUCCESS: 剩余油量 <= bingo油量
    - FAILURE: 剩余油量 > bingo油量
    """

    def update(self) -> Status:
        if self.agent.fuel <= self.env.options.aircraft_fuel_bingo_fuel:
            return Status.SUCCESS
        else:
            return Status.FAILURE
