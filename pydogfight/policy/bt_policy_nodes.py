from __future__ import annotations

import json

from pydogfight.core.models import BoundingBox
from pydogfight.utils.position_memory import PositionMemory
from pydogfight.policy.bt_policy import *
from pydogfight.core.world_obj import Aircraft, Missile
from pydogfight.core.actions import Actions
import pybts
from pybts import Status
import os

BASE_DIR = os.path.dirname(__file__)
DEFAULT_BT_GREEDY_POLICY_FILE = os.path.join(BASE_DIR, 'bt_greedy_policy.xml')


class InitGreedyPolicy(BTPolicyAction):
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

        test_agents = self.agent.generate_test_moves(
                angles=list(range(0, 360, 45)),
                dis=self.agent.turn_radius * 20
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


class GoHome(BTPolicyAction):
    """
    行为节点：返回基地
    此节点负责控制代理（如机器人或游戏角色）返回其起始基地或安全区域。此行为通常用于补给、避免危险或结束任务。

    - ALWAYS SUCCESS: 返回基地的操作被设计为总是成功，假定基地是可以到达的，并且代理具有返回基地的能力。
    """

    def update(self) -> Status:
        home_obj = self.env.get_home(self.agent.color)
        self.actions.put_nowait((Actions.go_to_location, home_obj.x, home_obj.y))
        return Status.SUCCESS


class IsFuelBingo(BTPolicyCondition):
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


class IsMissileDepleted(BTPolicyCondition):
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


class MissileThreatDetected(BTPolicyCondition):
    """
    条件节点:是否发现来袭导弹，如果发现了，可能需要进行规避
    - SUCCESS: 发现来袭导弹
    - FAILURE: 未发现
    """

    def update(self) -> Status:
        missiles = self.share_cache['missiles']
        if len(missiles) > 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class EnemyDetected(BTPolicyCondition):
    """
    行为节点：是否监测到敌机
    - SUCCESS: 监测到敌机
    - FAILURE: 没有监测到敌机
    """

    def update(self) -> Status:
        if 'nearest_enemy' not in self.share_cache:
            return Status.FAILURE
        if self.share_cache['nearest_enemy'] is None:
            return Status.FAILURE
        return Status.SUCCESS


class GoToCenter(BTPolicyAction):
    """
    行为节点：飞行到战场中心
    - ALWAYS SUCCESS
    """

    def update(self) -> Status:
        self.actions.put_nowait((Actions.go_to_location, 0, 0))
        return Status.SUCCESS

    @classmethod
    def creator(cls, d, c):
        return GoToCenter(name=d['name'])


class IsOnActiveRoute(BTPolicyCondition):
    """
    条件节点: 当前agent是否存在未完成的飞行路线
    - SUCCESS: 存在未完成的飞行路线
    - FAILURE: 不存在
    """

    def update(self) -> Status:
        if self.agent.route is not None:
            return Status.SUCCESS
        return Status.FAILURE


class IsInSafeArea(BTPolicyCondition):
    """
    条件节点: 是否处于安全区域，如果不在安全区域，则要往战场中心飞
    - SUCCESS: 处于安全区域
    - FAILURE: 不处于
    """

    def update(self) -> Status:
        bounds = BoundingBox.from_range(x_range=self.env.options.safe_x_range, y_range=self.env.options.safe_y_range)
        if bounds.contains(self.agent.location):
            return Status.SUCCESS
        return Status.FAILURE

    @classmethod
    def creator(cls, d, c):
        return IsInSafeArea(name=d['name'])


class EvadeMissile(BTPolicyAction):
    """
    行为节点: 规避导弹
    - SUCCESS: 规避成功
    - FAILURE: 规避失败（可能是不存在来袭导弹、无法规避导弹）
    """

    def update(self) -> Status:
        missiles = self.share_cache['missiles']
        test_agents = self.share_cache['test_agents']
        update_messages = [f'test_agents={len(test_agents)} missiles={len(missiles)}']
        if len(missiles) == 0:
            update_messages.append('No missiles')
            self.put_update_message(update_messages)
            return Status.FAILURE

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

            update_messages.append(f'max_under_hit_point {max_under_hit_point}')
            update_messages.append(f'go to location {go_to_location}')
            self.put_update_message(update_messages)
            return Status.SUCCESS

        self.put_update_message(update_messages)
        return Status.FAILURE

    @classmethod
    def creator(cls, d, c):
        return EvadeMissile(name=d['name'])


class AttackNearestEnemy(BTPolicyAction):
    """
    行为节点: 攻击最近的敌机（发射导弹）
    - SUCCESS: 发射导弹成功
    - FAILURE: 发射导弹失败（可能是没有发现最近敌机、剩余导弹为空、无法攻击到敌机）
    """

    def update(self) -> Status:
        update_messages = []
        if 'nearest_enemy' not in self.share_cache:
            update_messages.append('not found nearest_enemy')
            self.put_update_message(update_messages)
            return Status.FAILURE

        if self.agent.missile_count <= 0:
            update_messages.append('missile_count == 0')
            self.put_update_message(update_messages)
            return Status.FAILURE

        enemy = self.share_cache['nearest_enemy']
        hit_point = self.agent.predict_missile_intercept_point(target=enemy)

        update_messages.append(f'hit_point {hit_point}')

        if hit_point is not None and hit_point.time < 0.8 * self.env.options.missile_fuel_capacity / self.env.options.missile_fuel_consumption_rate:
            # 可以命中敌机
            self.actions.put_nowait((Actions.fire_missile, enemy.x, enemy.y))
            update_messages.append(f'可以命中敌机 fire missile {enemy.x} {enemy.y}')
            self.put_update_message(update_messages)
            return Status.SUCCESS
        return Status.FAILURE

    @classmethod
    def creator(cls, d, c):
        return AttackNearestEnemy(name=d['name'])


class PursueNearestEnemy(BTPolicyAction):
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

    def update(self) -> Status:
        update_messages = []
        if 'nearest_enemy' not in self.share_cache:
            update_messages.append('No nearest enemy')
            self.put_update_message(update_messages)
            return Status.FAILURE

        test_agents = self.share_cache['test_agents']
        enemy: Aircraft = self.share_cache['nearest_enemy']

        update_messages.append(f'test_agents={len(test_agents)} ')

        # 如果没有要躲避的任务，则尝试飞到更容易命中敌机，且更不容易被敌机命中的位置上（两者命中时间之差最大）
        max_time = -float('inf')
        go_to_location = None

        if enemy.distance(self.agent) > self.env.options.aircraft_radar_radius:
            # 超出雷达探测范围了，则朝着敌机的位置飞
            self.actions.put_nowait((Actions.go_to_location, enemy.x, enemy.y))
            update_messages.append('超出雷达探测范围了，则朝着敌机的位置飞')
            self.put_update_message(update_messages)
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
            self.put_update_message(update_messages)
            return Status.SUCCESS
        return Status.FAILURE


class Explore(BTPolicyAction):
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

    def update(self) -> Status:
        update_messages = []
        if self.agent.route is not None:
            update_messages.append('当前agent还有没有完成的路线')
            self.put_update_message(update_messages)
            return Status.FAILURE

        not_memory_pos = self.share_cache['memory'].pick_position()
        self.actions.put_nowait((Actions.go_to_location, not_memory_pos[0], not_memory_pos[1]))

        self.put_update_message(update_messages)
        return Status.SUCCESS


class KeepFly(BTPolicyAction):
    """
    行为节点保持当前航迹继续飞行
    """

    def update(self) -> Status:
        return Status.SUCCESS

    def to_data(self):
        return {
            'agent': self.agent.to_dict()
        }


class FollowRoute(BTPolicyAction):
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

    def update(self) -> Status:
        if self.route_index >= len(self.route):
            return Status.FAILURE

        pos = self.route[self.route_index]
        self.actions.put_nowait((Actions.go_to_location, pos[0], pos[1]))

        # 到了路由点附近
        if self.agent.distance(self.route[self.route_index]) <= self.agent.speed * self.env.options.delta_time * 2:
            self.route_index += 1
            if self.recursive:
                self.route_index %= len(self.route)

        return Status.SUCCESS

    def to_data(self):
        return {
            **super().to_data(),
            'route'      : [str(r) for r in self.route],
            'route_index': self.route_index,
            'recursive'  : self.recursive,
            'agent'      : self.agent.to_dict()
        }


class GoToLocation(BTPolicyAction):
    def __init__(self, x: float, y: float, name: str = ''):
        super().__init__(name=name)
        self.x = x
        self.y = y

    def update(self) -> Status:
        reach_distance = self.agent.speed * self.env.options.delta_time * self.env.options.reach_location_scale
        if self.agent.distance((self.x, self.y)) <= reach_distance:
            return Status.SUCCESS
        self.actions.put_nowait((Actions.go_to_location, self.x, self.y))
        return Status.RUNNING
    
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


class IsReachLocation(BTPolicyCondition):

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
        reach_distance = self.agent.speed * self.env.options.delta_time * self.env.options.reach_location_scale
        if self.agent.distance((self.x, self.y)) <= reach_distance:
            return Status.SUCCESS
        return Status.FAILURE


class BTPolicyBuilder(pybts.Builder):
    def __init__(self):
        super().__init__()
        self.register_greedy()

    def register_greedy(self):
        self.register_bt(
                InitGreedyPolicy,
                MissileThreatDetected,
                EnemyDetected,
                GoToCenter,
                IsInSafeArea,
                IsOnActiveRoute,
                EvadeMissile,
                AttackNearestEnemy,
                PursueNearestEnemy,
                Explore,
                GoHome,
                IsMissileDepleted,
                IsFuelBingo,
                FollowRoute,
                KeepFly,
                GoToLocation,
                IsReachLocation,
        )

    def build_default(self) -> pybts.Node:
        return self.build_from_file(DEFAULT_BT_GREEDY_POLICY_FILE)


def _main():
    builder = BTPolicyBuilder()
    tree = builder.build_from_file(DEFAULT_BT_GREEDY_POLICY_FILE)
    # print(pybts.utility.bt_to_xml(tree))
    with open(DEFAULT_BT_GREEDY_POLICY_FILE, 'r') as f:
        # print(pybts.utility.xml_to_json(f.read()))
        json_data = pybts.utility.xml_to_json(f.read())
        tree = builder.build_from_json(json_data=json_data)
        # print(tree)
        # print(pybts.utility.bt_to_xml(tree))
        for node in tree.iterate():
            print(node.name)


if __name__ == '__main__':
    # _main()
    builder = BTPolicyBuilder()
    with open(os.path.join(BASE_DIR, 'bt_policy_nodes.md'), 'w') as f:
        texts = ['## 战机策略行为树节点定义']
        for k, n in builder.repo_node.items():
            texts.append(f'**{k}**')
            for line in n.__doc__.split('\n'):
                line = line.strip()
                if line == '':
                    continue
                texts.append(line.strip())

        f.write('\n\n'.join(texts))
