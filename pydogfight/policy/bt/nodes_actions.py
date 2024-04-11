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


class GoHome(BTPolicyNode):
    """
    行为节点：返回基地
    此节点负责控制代理（如机器人或游戏角色）返回其起始基地或安全区域。此行为通常用于补给、避免危险或结束任务。

    - ALWAYS SUCCESS: 返回基地的操作被设计为总是成功，假定基地是可以到达的，并且代理具有返回基地的能力。
    """

    def updater(self) -> typing.Iterator[Status]:
        home_obj = self.env.battle_area.get_home(self.agent.color)
        yield from go_to_location_updater(self, home_obj.location)


class GoToCenter(BTPolicyNode):
    """
    行为节点：飞行到战场中心
    - ALWAYS SUCCESS
    """

    def updater(self) -> typing.Iterator[Status]:
        yield from go_to_location_updater(self, (0, 0))


class EvadeMissile(BTPolicyNode):
    """
    行为节点: 规避导弹
    - SUCCESS: 规避成功
    - FAILURE: 规避失败（可能是不存在来袭导弹、无法规避导弹）
    """

    def updater(self) -> typing.Iterator[Status]:
        # 获取导弹
        missiles = self.env.battle_area.detect_missiles(agent_name=self.agent_name, ignore_radar=False, only_enemy=True)
        if len(missiles) == 0:
            yield Status.FAILURE
            return

        test_agents = self.agent.generate_test_moves(in_safe_area=True)
        # 从周围N个点中寻找一个能够让导弹飞行时间最长且自己飞行时间最短的点来飞 （导弹飞行时间 - 自己飞行时间）最大
        max_diff_time = 0
        go_to_location = None
        for agent_tmp in test_agents:
            diff_time = 0
            for mis in missiles[:1]:  # 只规避最近的导弹
                under_hit_point = mis.predict_aircraft_intercept_point(target=agent_tmp)
                optimal_path = calc_optimal_path(
                        start=self.agent.waypoint,
                        target=agent_tmp.location,
                        turn_radius=self.agent.turn_radius,
                )
                if under_hit_point is None:
                    continue
                if under_hit_point.time == float('inf'):
                    under_hit_point.time = 10000

                diff_time += under_hit_point.time - optimal_path.length / self.agent.speed

            if diff_time > max_diff_time:
                max_diff_time = diff_time
                go_to_location = agent_tmp.location

        if go_to_location is None:
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, go_to_location)


class FireMissileAtNearestEnemy(BTPolicyNode):
    """
    行为节点: 朝着最近的敌机发射导弹
    - SUCCESS: 发射导弹成功
    - FAILURE: 发射导弹失败（可能是没有发现最近敌机、剩余导弹为空、无法攻击到敌机）
    """

    def updater(self) -> typing.Iterator[Status]:
        if self.agent.missile_count <= 0:
            self.put_update_message('Missile Count == 0')
            yield Status.FAILURE
            return
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        hit_point = self.agent.predict_missile_intercept_point(target=enemy)

        self.put_update_message(f'hit_point {hit_point}')

        if hit_point is not None and hit_point.time < 0.8 * self.env.options.missile_fuel_capacity / self.env.options.missile_fuel_consumption_rate:
            # 有可能命中敌机
            self.actions.put_nowait((Actions.fire_missile, enemy.x, enemy.y))
            self.put_update_message(f'可以命中敌机 fire missile {enemy.x} {enemy.y}')
            # 每隔N秒最多发射一枚导弹
            yield from delay_updater(env=self.env, time=self.env.options.missile_fire_interval, status=Status.SUCCESS)
        else:
            yield Status.FAILURE




class Explore(BTPolicyNode):
    """
    行为节点：探索未知区域
    此节点负责控制代理（如机器人或游戏角色）探索它尚未访问的地方。节点的目标是扩大代理的知识范围，通过探索环境来发现新的区域或点。

    - SUCCESS: 探索成功，表示代理成功移动到一个之前未探索的位置。
    - FAILURE: 探索失败，可能因为以下原因：
        - 代理当前有尚未完成的路线或任务，因此无法开始新的探索。
        - 环境中没有更多的未探索区域，或者无法从当前位置移动到未探索的区域。
    """

    def updater(self) -> typing.Iterator[Status]:
        if self.agent.route is not None:
            self.put_update_message('当前agent还有没有完成的路线')
            yield Status.RUNNING
        go_to_location = self.agent.position_memory.pick_position()
        yield from go_to_location_updater(self, go_to_location)


class KeepFly(BTPolicyNode):
    """
    行为节点保持当前航迹继续飞行
    """

    def update(self) -> Status:
        return Status.SUCCESS


class FollowRoute(BTPolicyNode):
    """
    沿着一条预设的航线飞行
    """

    def __init__(self, route: list | str, recursive: bool | str = False, **kwargs):
        super().__init__(**kwargs)
        self.route: list = self.converter.list(route)
        self.route_index = 0
        self.recursive = self.converter.bool(recursive)
        assert len(self.route) > 0

    def updater(self) -> typing.Iterator[Status]:
        while self.recursive:
            for index in range(len(self.route)):
                self.route_index = index
                yield from go_to_location_updater(self, self.route[index])

    def to_data(self):
        return {
            **super().to_data(),
            'route'      : [str(r) for r in self.route],
            'route_index': self.route_index,
            'recursive'  : self.recursive,
            'agent'      : self.agent.to_dict()
        }


class GoToLocation(BTPolicyNode):
    def __init__(self, x: float | str, y: float | str, **kwargs):
        super().__init__(**kwargs)
        self.x = self.converter.float(x)
        self.y = self.converter.float(y)

    def initialise(self) -> None:
        super().initialise()
        self.actions.put_nowait((Actions.go_to_location, self.x, self.y))

    def updater(self) -> typing.Iterator[Status]:
        yield from go_to_location_updater(self, (self.x, self.y))

    def to_data(self):
        return {
            **super().to_data(),
            'x': self.x,
            'y': self.y,
        }

# TODO: 条件节点，导弹命中敌机，需要考虑一些匹配
