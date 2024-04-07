from __future__ import annotations
from collections import defaultdict
import random
from pydogfight.policy.policy import AgentPolicy
from pydogfight.envs import Dogfight2dEnv
from pydogfight.core.world_obj import Missile, Aircraft
from pydogfight.core.actions import Actions


class GreedyPolicy(AgentPolicy):
    """
    贪心策略，追逐最近的敌人
    """

    def __init__(self, env: Dogfight2dEnv, agent_name: str, update_interval: float = 1):
        super().__init__(env=env, agent_name=agent_name, update_interval=update_interval)

        self.memory_gap = int(self.options.aircraft_speed * 10)
        self.safe_boundary_distance = self.options.aircraft_speed * 20  # 距离边界的安全距离
        self.max_center_distance = min(self.options.game_size) / 2 - self.safe_boundary_distance
        self.safe_x_range = [-int(self.options.game_size[0] / 2 + self.safe_boundary_distance),
                             int(self.options.game_size[0] / 2 - self.safe_boundary_distance)]
        self.safe_y_range = [-int(self.options.game_size[1] / 2 + self.safe_boundary_distance),
                             int(self.options.game_size[1] / 2 - self.safe_boundary_distance)]

        self.enemy: Aircraft | None = None
        self.memory = defaultdict(int)

    def reset_memory(self):
        self.memory = defaultdict(int)
        for x in range(self.safe_x_range[0],
                       self.safe_x_range[1],
                       self.memory_gap):
            for y in range(self.safe_y_range[0],
                           self.safe_y_range[1],
                           self.memory_gap):
                key = self._memory_key(x=x, y=y)
                self.memory[key] = 0

    def _memory_key(self, x: float, y: float) -> tuple[float, float]:
        if x < self.safe_x_range[0]:
            x = self.safe_x_range[0]
        if x > self.safe_x_range[1]:
            x = self.safe_x_range[1]
        if y < self.safe_y_range[0]:
            y = self.safe_y_range[0]
        if y > self.safe_y_range[1]:
            y = self.safe_y_range[1]
        return int(x / self.memory_gap), int(y / self.memory_gap)

    def find_not_memory_pos(self):
        min_value = min(self.memory.values())
        min_pos = []
        for pos in self.memory:
            if self.memory[pos] == min_value:
                min_pos.append(pos)
        min_pos = random.choice(min_pos)
        return min_pos[0] * self.memory_gap, min_pos[1] * self.memory_gap

    def reset(self):
        super().reset()
        self.enemy: Aircraft | None = None
        self.memory = defaultdict(int)
        self.reset_memory()

    def execute(self, observation, delta_time: float):
        agent = self.agent
        missiles = []

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
                # hit_point = obj.predict_aircraft_intercept_point(
                #         target=agent,
                # )
                # if hit_point is not None and hit_point.time <= 15:
                #     missiles.append(obj)

        go_to_location = None
        fire_missile = None

        MAX_CENTER_DISTANCE = min(self.options.game_size) / 2 - self.safe_boundary_distance  # 建议距离战场中心最远距离
        test_agents = agent.generate_test_moves(
                in_safe_area=True
        )

        test_agents = list(filter(lambda agent: agent.distance((0, 0)) < MAX_CENTER_DISTANCE, test_agents))

        # 如果距离战场中心过远，则往战场中心飞
        if go_to_location is None and agent.distance((0, 0)) > MAX_CENTER_DISTANCE:
            go_to_location = (0, 0)

        # 如果发现了导弹，则优先规避导弹
        if len(missiles) > 0 and go_to_location is None:
            # 从周围8个点中寻找一个能够让导弹飞行时间最长的点来飞
            max_under_hit_point = None
            for agent_tmp in test_agents:
                for mis in missiles:
                    under_hit_point = mis.predict_aircraft_intercept_point(target=agent_tmp)
                    if under_hit_point is None:
                        continue
                    if max_under_hit_point is None or under_hit_point.time > max_under_hit_point.time:
                        max_under_hit_point = under_hit_point
                        go_to_location = agent_tmp.location

        enemy = self.env.battle_area.find_nearest_enemy(agent_name=self.agent_name, ignore_radar=False)
        if enemy is not None:
            self.enemy = enemy.__copy__()
        elif self.enemy is not None:
            self.enemy.move_forward(delta_time=delta_time)  # 假设之前发现的敌方按照原来的路线继续飞行
        # 如果发现敌方，则朝敌方发射导弹
        if enemy is not None:
            hit_point = agent.predict_missile_intercept_point(target=enemy)
            if hit_point is not None and hit_point.time < 0.8 * self.options.missile_fuel_capacity / self.options.missile_fuel_consumption_rate:
                # 可以命中敌机
                fire_missile = enemy.location

            if go_to_location is None:
                # 如果没有要躲避的任务，则尝试飞到更容易命中敌机，且更不容易被敌机命中的位置上（两者命中时间之差最大）
                max_time = None
                ATTACK_RATIO = 0.5  # 0.5表示 进攻和规避的系数相同，越接近1代表越倾向于进攻，1代表完全放弃规避，0代表完全放弃进攻
                for agent_tmp in test_agents:
                    hit_point = agent_tmp.predict_missile_intercept_point(target=enemy)
                    under_hit_point = enemy.predict_missile_intercept_point(target=agent_tmp)
                    time_tmp = hit_point.time * ATTACK_RATIO - under_hit_point.time * (1 - ATTACK_RATIO)
                    if max_time is None or max_time < time_tmp:
                        max_time = time_tmp
                        go_to_location = agent_tmp.location
        elif self.enemy is not None and go_to_location is None:
            go_to_location = self.enemy.location

        # 记忆走过的地方
        self.memory[self._memory_key(x=agent.x, y=agent.y)] += 1

        # if go_to_location is None:
        if go_to_location is None and agent.route is None:
            # 随机找个没去过的地方
            go_to_location = self.find_not_memory_pos()

        if go_to_location is not None:
            self.actions.put_nowait([Actions.go_to_location, go_to_location[0], go_to_location[1]])

        if fire_missile is not None:
            self.actions.put_nowait([Actions.fire_missile, fire_missile[0], fire_missile[1]])

        return
