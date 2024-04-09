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
        yield Status.SUCCESS


class GoToCenter(BTPolicyNode):
    """
    行为节点：飞行到战场中心
    - ALWAYS SUCCESS
    """

    def updater(self) -> typing.Iterator[Status]:
        yield from go_to_location_updater(self, (0, 0))
        yield Status.SUCCESS


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
        yield Status.SUCCESS


class AttackNearestEnemy(BTPolicyNode):
    """
    行为节点: 攻击最近的敌机（发射导弹）
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
            # 每隔3s最多发射一枚导弹
            yield from delay_updater(env=self.env, time=self.env.options.missile_fire_interval, status=Status.SUCCESS)
        else:
            yield Status.FAILURE


class GoToNearestEnemy(BTPolicyNode):
    """
    飞到最近的敌机的位置
    """

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, enemy.location)
        yield Status.SUCCESS
        return


class AwayFromNearestEnemy(BTPolicyNode):
    """
    远离最近的敌机
    """

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        agent = self.agent
        vec = (agent.x - enemy.x, agent.y - enemy.y)
        target = (agent.x + vec[0], agent.y + vec[1])
        yield from go_to_location_updater(self, target)
        yield Status.SUCCESS
        return


class PursueNearestEnemy(BTPolicyNode):
    """
    行为节点：追击最近的敌机
    此节点用于判断并执行对最近敌机的追击动作。它会评估并选择一个位置，该位置既能提高对敌机的命中率，又能降低被敌机命中的风险。

    - SUCCESS: 追击成功，表示本节点控制的飞机成功调整了位置，优化了对最近敌机的攻击角度或位置。
    - FAILURE: 追击失败，可能因为以下原因：
        - 未能检测到最近的敌机，可能是因为敌机超出了雷达的探测范围。
        - 未能计算出一个更有利的位置，或者在当前情况下调整位置不会带来明显的优势。
        - 其他因素，如环境变量、飞机状态或策略设置，导致追击行动无法执行。

    Parameters:
    - name (str): 此行为节点的名称。
    - attack_ratio (float): 进攻比例，用于决定在追击过程中进攻的倾向性，较高值意味着更偏向于进攻。
    - evade_ratio (float): 逃避比例，用于决定在追击过程中防御的倾向性，较高值意味着更偏向于防御。
    """

    def __init__(self, attack_ratio: float | str = 0.5, evade_ratio: float | str = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.attack_ratio = float(attack_ratio)
        self.evade_ratio = float(evade_ratio)

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        # 如果没有要躲避的任务，则尝试飞到更容易命中敌机，且更不容易被敌机命中的位置上（两者命中时间之差最大）
        min_time = float('inf')
        go_to_location = None

        if enemy.distance(self.agent) > self.env.options.aircraft_radar_radius:
            # 超出雷达探测范围了，则朝着敌机的位置飞
            go_to_location = (enemy.x, enemy.y)
            self.put_update_message('超出雷达探测范围了，则朝着敌机的位置飞')
        else:
            test_agents = self.agent.generate_test_moves(
                    in_safe_area=True
            )
            self.put_update_message(f'test_agents={len(test_agents)} ')
            for agent_tmp in test_agents:
                assert isinstance(agent_tmp, Aircraft)
                hit_point = agent_tmp.predict_missile_intercept_point(target=enemy)  # 我方命中敌机
                under_hit_point = enemy.predict_missile_intercept_point(target=agent_tmp)  # 敌机命中我方
                if hit_point.time == float('inf'):
                    hit_point.time = 10000
                if under_hit_point.time == float('inf'):
                    under_hit_point.time = 10000
                time_tmp = hit_point.time * self.attack_ratio - under_hit_point.time * self.evade_ratio  # 让我方命中敌机的时间尽可能小，敌方命中我方的时间尽可能大
                if time_tmp < min_time:
                    min_time = time_tmp
                    go_to_location = agent_tmp.location

        if go_to_location is None:
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, go_to_location)
        yield Status.SUCCESS
        return


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
        yield Status.SUCCESS


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
        yield Status.SUCCESS

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
        yield Status.SUCCESS

    def to_data(self):
        return {
            **super().to_data(),
            'x': self.x,
            'y': self.y,
        }

# TODO: 条件节点，导弹命中敌机，需要考虑一些匹配
