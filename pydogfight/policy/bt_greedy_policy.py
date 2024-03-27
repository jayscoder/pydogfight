from __future__ import annotations
import math
from pydogfight.policy.bt_policy import *
from pydogfight.core.actions import Actions
from pydogfight import *
import numpy as np
from queue import Queue
from collections import defaultdict
import random
from pydogfight.policy.policy import AgentPolicy
import py_trees
from py_trees.common import Status
from pydogfight.core.models import BoundingBox
from pydogfight.bt import BTXMLBuilder
from pydogfight.utils.position_memory import PositionMemory


class InitGreedy(BTPolicyAction):

    def __init__(self, name: str, memory_sep: int):
        super().__init__(name=name)
        self.memory_sep = memory_sep

    @classmethod
    def creator(cls, d, c):
        return InitGreedy(name=d['name'], memory_sep=int(d.get('memory_sep', 1000)))

    def update(self) -> Status:
        nearest_enemy = self.env.battle_area.find_nearest_enemy(agent_name=self.agent_name,
                                                                ignore_radar=False)
        if nearest_enemy is not None:
            self.share_cache['nearest_enemy'] = nearest_enemy.__copy__()

        test_agents = self.agent.generate_test_moves(
                angles=list(range(0, 360, 45)),
                dis=self.agent.turn_radius * 10
        )

        test_agents = list(
                filter(lambda agent: self.env.options.safe_boundary.contains(agent.location), test_agents))
        self.share_cache['test_agents'] = test_agents

        missiles = []
        agent = self.env.get_agent(self.agent_name)
        for obj in self.env.battle_area.objs.values():
            if obj.name == self.agent_name:
                continue
            if not agent.in_radar_range(obj) or obj.destroyed:
                # 不在雷达范围内或者已经被摧毁了
                continue
            elif isinstance(obj, Missile):
                if obj.color == agent.color:
                    continue
                missiles.append(obj)
        self.share_cache['missiles'] = missiles

        # 记忆走过的地方
        self.share_cache['memory'].add_position(self.agent.location)
        return Status.SUCCESS

    def reset(self):
        super().reset()
        self.share_cache['memory'] = PositionMemory(boundary=self.env.options.safe_boundary, sep=self.memory_sep)


class HasMissiles(BTPolicyAction):
    def update(self) -> Status:
        missiles = self.share_cache['missiles']
        if len(missiles) > 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE

    @classmethod
    def creator(cls, d, c):
        return HasMissiles(name=d['name'])


class GoToCenter(BTPolicyAction):
    def update(self) -> Status:
        self.actions.put_nowait((Actions.go_to_location, 0, 0))
        return Status.SUCCESS

    @classmethod
    def creator(cls, d, c):
        return GoToCenter(name=d['name'])


class InSafeArea(BTPolicyCondition):
    def update(self) -> Status:
        bounds = BoundingBox.from_range(x_range=self.env.options.safe_x_range, y_range=self.env.options.safe_y_range)
        if bounds.contains(self.agent.location):
            return Status.SUCCESS
        return Status.FAILURE

    @classmethod
    def creator(cls, d, c):
        return InSafeArea(name=d['name'])


class EvadeMissile(BTPolicyAction):

    def update(self) -> Status:
        missiles = self.share_cache['missiles']
        if len(missiles) == 0:
            return Status.FAILURE
        test_agents = self.share_cache['test_agents']
        # 从周围8个点中寻找一个能够让导弹飞行时间最长的点来飞
        max_under_hit_point = None
        go_to_location = None
        for agent_tmp in test_agents:
            for mis in missiles:
                under_hit_point = mis.predict_aircraft_intercept_point(target=agent_tmp)
                if under_hit_point is None:
                    continue
                if max_under_hit_point is None or under_hit_point.time > max_under_hit_point.time:
                    max_under_hit_point = under_hit_point
                    go_to_location = agent_tmp.location

        if go_to_location is not None:
            self.actions.put_nowait((Actions.go_to_location, go_to_location[0], go_to_location[1]))
            return Status.SUCCESS
        else:
            return Status.FAILURE

    @classmethod
    def creator(cls, d, c):
        return EvadeMissile(name=d['name'])


class AttackNearestEnemy(BTPolicyAction):

    def update(self) -> Status:
        if 'nearest_enemy' not in self.share_cache:
            return Status.FAILURE
        enemy = self.share_cache['nearest_enemy']
        hit_point = self.agent.predict_missile_intercept_point(target=enemy)
        if hit_point is not None and hit_point.time < 0.8 * self.env.options.missile_fuel_capacity / self.env.options.missile_fuel_consumption_rate:
            # 可以命中敌机
            self.actions.put_nowait((Actions.fire_missile, enemy.x, enemy.y))
        else:
            return Status.FAILURE
        # if go_to_location is None:
        #

        return Status.SUCCESS

    @classmethod
    def creator(cls, d, c):
        return AttackNearestEnemy(name=d['name'])


class PursueNearestEnemy(BTPolicyAction):

    def __init__(self, name: str, attack_ratio: float = 0.5, evade_ratio: float = 0.5):
        super().__init__(name=name)
        self.attack_ratio = attack_ratio
        self.evade_ratio = evade_ratio

    @classmethod
    def creator(cls, d, c):
        return PursueNearestEnemy(
                name=d['name'],
                attack_ratio=d.get('attack_ratio', 0.5),
                evade_ratio=d.get('evade_ratio', 0.5))

    def update(self) -> Status:
        if 'nearest_enemy' not in self.share_cache:
            return Status.FAILURE

        test_agents = self.share_cache['test_agents']
        enemy: Aircraft = self.share_cache['nearest_enemy']
        # 如果没有要躲避的任务，则尝试飞到更容易命中敌机，且更不容易被敌机命中的位置上（两者命中时间之差最大）
        max_time = -float('inf')
        go_to_location = None

        if enemy.distance(self.agent) > self.env.options.aircraft_radar_radius:
            # 超出雷达探测范围了，则朝着敌机的位置飞
            self.actions.put_nowait((Actions.go_to_location, enemy.x, enemy.y))
            return Status.SUCCESS

        for agent_tmp in test_agents:
            hit_point = agent_tmp.predict_missile_intercept_point(target=enemy)
            under_hit_point = enemy.predict_missile_intercept_point(target=agent_tmp)
            if hit_point.time == float('inf'):
                hit_point.time = 10000
            if under_hit_point.time == float('inf'):
                under_hit_point.time = 10000
            time_tmp = hit_point.time * self.attack_ratio - under_hit_point.time * self.evade_ratio
            if max_time < time_tmp:
                max_time = time_tmp
                go_to_location = agent_tmp.location

        if go_to_location is not None:
            self.actions.put_nowait((Actions.go_to_location, go_to_location[0], go_to_location[1]))
            return Status.SUCCESS
        else:
            return Status.FAILURE


class Explore(BTPolicyAction):
    """
    探索没去过的敌方
    """

    @classmethod
    def creator(cls, d, c):
        return Explore(
                name=d['name'])

    def update(self) -> Status:
        if self.agent.route is not None:
            return Status.FAILURE
        not_memory_pos = self.share_cache['memory'].pick_position()
        self.actions.put_nowait((Actions.go_to_location, not_memory_pos[0], not_memory_pos[1]))
        return Status.SUCCESS


class BTGreedyBuilder(BTXMLBuilder):
    def __init__(self):
        super().__init__()
        self.register_greedy()

    def register_greedy(self):
        self.register('InitGreedy', InitGreedy.creator)
        self.register('HasMissiles', HasMissiles.creator)
        self.register('GoToCenter', GoToCenter.creator)
        self.register('InSafeArea', InSafeArea.creator)
        self.register('EvadeMissile', EvadeMissile.creator)
        self.register('AttackNearestEnemy', AttackNearestEnemy.creator)
        self.register('PursueNearestEnemy', PursueNearestEnemy.creator)
        self.register('Explore', Explore.creator)


BT_GREEDY_XML = """
<Sequence>
    <InitGreedy memory_gap="1000"/>
    <Parallel>
        <AttackNearestEnemy/>
        <Selector>
            <Sequence>
                <InSafeArea/>
                <Selector>
                    <EvadeMissile/>
                    <PursueNearestEnemy/>
                    <Explore/>
                </Selector>
            </Sequence>
            <GoToCenter/>
        </Selector>
    </Parallel>
</Sequence>
"""


def _main():
    builder = BTGreedyBuilder()
    tree = builder.build_from_xml_text(BT_GREEDY_XML)


if __name__ == '__main__':
    _main()
