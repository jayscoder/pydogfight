from __future__ import annotations

import json
import typing

from pydogfight.core.models import BoundingBox
from pydogfight.utils.position_memory import PositionMemory
from pydogfight.policy.bt.nodes import *
from pydogfight.core.world_obj import Aircraft, Missile
from pydogfight.algos.traj import calc_optimal_path
from pydogfight.core.actions import Actions
import pybts
from pybts import Status
import os

from pydogfight.policy.bt.common import *


class InitGreedyPolicy(BTPolicyNode):
    """
    行为节点：初始化贪心策略
    此节点用于在决策过程开始时初始化贪心策略，设置必要的环境和参数，以便后续节点可以执行基于贪心逻辑的决策。它主要负责收集和初始化与战斗相关的数据，如敌人的位置、可用的导弹等信息。

    - SUCCESS: 初始化成功，表示贪心策略相关的数据和参数已经准备完毕，可以开始执行具体的贪心决策。
    - FAILURE: 初始化失败，通常不应发生，除非在获取或设置初始数据时出现问题。

    Parameters:
    - name (str): 此行为节点的名称。
    - memory_sep (int): 用于分割记忆中的位置，以帮助管理探索的区域。

    备注：
    - 此节点应在决策树执行的早期阶段运行，以确保后续的行为节点可以访问到初始化后的数据和状态。
    - 使用共享缓存来存储和传递初始化的数据，确保决策树中的其他节点可以访问这些信息。
    """

    def __init__(self, name: str, memory_sep: int = 1000):
        super().__init__(name=name)
        self.memory_sep = memory_sep

    def to_data(self):
        nearest_enemy = self.share_cache.get('nearest_enemy', None)
        if nearest_enemy is not None:
            nearest_enemy = nearest_enemy.to_dict()
        missiles = self.share_cache.get('missiles', None)
        if missiles is not None:
            missiles = len(missiles)
        agent = self.agent
        if agent is not None:
            agent = agent.to_dict()
        return {
            **super().to_data(),
            'memory_sep'   : self.memory_sep,
            'nearest_enemy': nearest_enemy,
            'missiles'     : missiles,
            'agent'        : agent
        }

    @classmethod
    def creator(cls, d, c):
        return InitGreedyPolicy(name=d['name'], memory_sep=int(d.get('memory_sep', 1000)))

    def update(self) -> Status:
        nearest_enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)
        if nearest_enemy is not None:
            self.share_cache['nearest_enemy'] = nearest_enemy.__copy__()

        self.share_cache['test_agents'] = self.agent.generate_test_moves(
                in_safe_area=True
        )

        self.share_cache['missiles'] = self.env.battle_area.detect_missiles(agent_name=self.agent_name)
        # 记忆走过的地方
        self.share_cache['memory'].add_position(self.agent.location)
        return Status.SUCCESS

    def reset(self):
        super().reset()
        self.share_cache['memory'] = PositionMemory(boundary=self.env.options.safe_boundary(), sep=self.memory_sep)


class GoHome(BTPolicyNode):
    """
    行为节点：返回基地
    此节点负责控制代理（如机器人或游戏角色）返回其起始基地或安全区域。此行为通常用于补给、避免危险或结束任务。

    - ALWAYS SUCCESS: 返回基地的操作被设计为总是成功，假定基地是可以到达的，并且代理具有返回基地的能力。
    """

    def updater(self) -> typing.Iterator[Status]:
        home_obj = self.env.get_home(self.agent.color)
        yield from go_to_location_updater(self, home_obj.location)
        yield Status.SUCCESS


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


class MissileThreatDetected(BTPolicyNode, pybts.Condition):
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


class EnemyDetected(BTPolicyNode, pybts.Condition):
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


class GoToCenter(BTPolicyNode):
    """
    行为节点：飞行到战场中心
    - ALWAYS SUCCESS
    """

    def updater(self) -> typing.Iterator[Status]:
        yield from go_to_location_updater(self, (0, 0))
        yield Status.SUCCESS

    @classmethod
    def creator(cls, d, c):
        return GoToCenter(name=d['name'])


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

    @classmethod
    def creator(cls, d, c):
        return IsInSafeArea(name=d['name'])


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
            for mis in missiles[:1]: # 只规避最近的导弹
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

    @classmethod
    def creator(cls, d, c):
        return EvadeMissile(name=d['name'])


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

    @classmethod
    def creator(cls, d, c):
        return AttackNearestEnemy(name=d['name'])


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


class IsNearEnemy(BTPolicyNode):
    """
    是否靠近敌机
    """

    def __init__(self, radar_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.radar_ratio = radar_ratio  # 以自己的雷达半径为判断依据

    def update(self) -> Status:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        distance = self.agent.distance(enemy)

        if distance <= self.radar_ratio * self.agent.radar_radius:
            return Status.SUCCESS
        else:
            return Status.FAILURE


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

    def __init__(self, name: str = '', attack_ratio: float = 0.5, evade_ratio: float = 0.5):
        super().__init__(name=name)
        self.attack_ratio = attack_ratio
        self.evade_ratio = evade_ratio

    @classmethod
    def creator(cls, d, c):
        return PursueNearestEnemy(
                name=d['name'],
                attack_ratio=float(d.get('attack_ratio', 0.5)),
                evade_ratio=float(d.get('evade_ratio', 0.5)),
        )

    def to_data(self):
        return {
            **super().to_data(),
            'attack_ratio': self.attack_ratio,
            'evade_ratio' : self.evade_ratio
        }

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

    @classmethod
    def creator(cls, d, c):
        return Explore(
                name=d['name'])

    def updater(self) -> typing.Iterator[Status]:
        if self.agent.route is not None:
            self.put_update_message('当前agent还有没有完成的路线')
            yield Status.RUNNING
        go_to_location = self.share_cache['memory'].pick_position()
        yield from go_to_location_updater(self, go_to_location)
        yield Status.SUCCESS


class KeepFly(BTPolicyNode):
    """
    行为节点保持当前航迹继续飞行
    """

    def update(self) -> Status:
        return Status.SUCCESS

    def to_data(self):
        return {
            'agent': self.agent.to_dict()
        }


class FollowRoute(BTPolicyNode):
    """
    沿着一条预设的航线飞行
    """

    def __init__(self, route: list, recursive: bool = False, name: str = ''):
        super().__init__(name=name)
        self.route = route
        self.route_index = 0
        self.recursive = recursive
        assert len(self.route) > 0

    @classmethod
    def creator(cls, d: dict, c: list):
        return cls(
                name=d['name'],
                route=json.loads(d['route']),
                recursive=bool(d.get('recursive', False))
        )

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
    def __init__(self, x: float, y: float, name: str = ''):
        super().__init__(name=name)
        self.x = x
        self.y = y

    def initialise(self) -> None:
        super().initialise()
        self.actions.put_nowait((Actions.go_to_location, self.x, self.y))

    def updater(self) -> typing.Iterator[Status]:
        yield from go_to_location_updater(self, (self.x, self.y))
        yield Status.SUCCESS

    @classmethod
    def creator(cls, d: dict, c: list):
        return GoToLocation(
                x=float(d['x']), y=float(d['y']), name=d['name']
        )

    def to_data(self):
        return {
            **super().to_data(),
            'x': self.x,
            'y': self.y,
        }


class IsReachLocation(BTPolicyNode, pybts.Condition):
    """由于策略的更新时间较长，可能无法正确判断是否到达某个目标点"""

    def __init__(self, x: float, y: float, name: str = ''):
        super().__init__(name=name)
        self.x = x
        self.y = y

    @classmethod
    def creator(cls, d: dict, c: list):
        return IsReachLocation(
                x=float(d['x']), y=float(d['y']), name=d['name']
        )

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

# TODO: 条件节点，导弹命中敌机，需要考虑一些匹配