from __future__ import annotations

from pydogfight.utils.intercept import *
from pydogfight.utils.traj import calc_optimal_path
from pydogfight.policy.bt.common import *


# class BasePursueEnemyNode(BTPolicyNode):
#
#     def lag_pursue_waypoint(self, enemy: Aircraft, lag_time: float):
#         distance = self.agent.distance(enemy)
#         use_lag_time = max(distance / self.agent.speed, lag_time)
#         return enemy.waypoint.move(-self.agent.speed * use_lag_time)  # 飞到敌机之前的位置
#
#     @abstractmethod
#     def gen_location(self, enemy: Aircraft) -> tuple[float, float]:
#         raise NotImplemented
#
#     def updater(self) -> typing.Iterator[Status]:
#         enemy = self.env.battle_area.find_nearest_enemy(
#                 agent_name=self.agent_name,
#                 ignore_radar=False)
#         if enemy is None:
#             self.put_update_message('No nearest enemy')
#             yield Status.FAILURE
#             return
#
#         yield from go_to_location_updater(self, self.gen_location(enemy=enemy))
#         yield Status.SUCCESS


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
            test_waypoints = self.agent.generate_test_moves(
                    in_safe_area=True
            )
            for waypoint in test_waypoints:
                hit_point = optimal_predict_intercept_point(
                        self_speed=self.env.options.missile_speed,
                        self_wpt=waypoint,
                        self_turn_radius=self.env.options.missile_min_turn_radius,
                        target_wpt=enemy.waypoint,
                        target_speed=enemy.speed,
                )  # 我方的导弹命中敌机

                under_hit_point = enemy.predict_missile_intercept_point(
                        target_wpt=waypoint,
                        target_speed=enemy.speed)  # 敌机的导弹命中我方

                if hit_point.time == float('inf'):
                    hit_point.time = 10000
                if under_hit_point.time == float('inf'):
                    under_hit_point.time = 10000
                time_tmp = hit_point.time * self.attack_ratio - under_hit_point.time * self.evade_ratio  # 让我方命中敌机的时间尽可能小，敌方命中我方的时间尽可能大
                if time_tmp < min_time:
                    min_time = time_tmp
                    go_to_location = waypoint.location

        if go_to_location is None:
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, go_to_location)


class AwayFromNearestEnemy(BTPolicyNode):
    """
    远离最近的敌机
    """

    def __init__(self, distance: float | str = 2000, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return
        move_d = self.converter.float(self.distance)
        new_wpt = self.agent.waypoint.move_towards(target=enemy.waypoint.location, d=-move_d)  # 负数表示远离
        yield from go_to_location_updater(self, location=new_wpt.location)


class GoToNearestEnemy(BTPolicyNode):
    """
    飞到最近的敌机的位置
    别名：PurePursueNearestEnemy
    """

    def __init__(self, distance: float | str = 2000, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    @classmethod
    def calculate_location(cls, enemy: Aircraft):
        return enemy.waypoint.location

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return
        move_d = self.converter.float(self.distance)
        new_wpt = self.agent.waypoint.move_towards(target=enemy.waypoint.location, d=move_d, allow_over=False)
        yield from go_to_location_updater(self, location=new_wpt.location)


class PurePursueNearestEnemy(GoToNearestEnemy):
    """
    Pure模式追逐最近的敌机
    飞到敌机所在的位置
    """
    pass


class FPolePursueNearestEnemy(BTPolicyNode):
    """
    F-Pole模式追逐最近的敌机
    朝着让目标在你左右两侧30度的方向飞
    """

    def __init__(self, angle: str | float = 30, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.angle = self.converter.float(self.angle)

    def to_data(self):
        return {
            **super().to_data(),
            'angle': self.angle
        }

    @classmethod
    def calculate_location(cls, agent: Aircraft, enemy: Aircraft, angle: float = 30) -> tuple[float, float]:
        rel_wpt = agent.waypoint.relative_polar_waypoint(enemy.waypoint)
        if rel_wpt.theta > 0:
            intercept_heading = rel_wpt.theta - angle
        else:
            intercept_heading = rel_wpt.theta + angle

        target_wpt = agent.waypoint.move(agent.speed * 10, angle=intercept_heading)

        return target_wpt.location

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)

        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, self.calculate_location(
                agent=self.agent,
                enemy=enemy,
                angle=self.angle
        ))


class LeadPursueNearestEnemy(BTPolicyNode):
    """
    Lead pursue nearest enemy
    如果未来会相交，则朝相交点飞行，如果不会相交，则朝着敌机15s后的预测位置飞
    """

    def __init__(self, predict_time: float | str = 15, **kwargs):
        super().__init__(**kwargs)
        self.predict_time = predict_time

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.predict_time = self.converter.float(self.predict_time)

    def to_data(self):
        return {
            **super().to_data(),
            'predict_time': self.predict_time
        }

    @classmethod
    def calculate_location(cls, agent: Aircraft, enemy: Aircraft, predict_time: float = 15) -> tuple[float, float]:
        # 计算拦截目标点
        intercept_point = agent.predict_intercept_point(
                target_wpt=enemy.waypoint,
                target_speed=enemy.speed
        )

        if intercept_point is not None:
            return intercept_point.point
        else:
            # 朝着敌机15s后预测的位置飞
            target_wpt = enemy.waypoint.move(d=enemy.speed * predict_time)
            return target_wpt.location

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)
        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, self.calculate_location(
                predict_time=self.predict_time,
                agent=self.agent,
                enemy=enemy
        ))


class LagPursueNearestEnemy(BTPolicyNode):
    """
    https://www.combataircraft.com/en/Tactics/Air-To-Air/Lag-Pursuit/
    飞到敌机一段时间之前的位置，这个时间由双方距离动态确定，距离敌机越远，则越接近15s，距离敌机越近，则越接近0

    lag_time: 滞后时间

    """

    def __init__(self, lag_time: float | str = 15, **kwargs):
        super().__init__(**kwargs)
        self.lag_time = lag_time

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.lag_time = self.converter.float(self.lag_time)

    def to_data(self):
        return {
            **super().to_data(),
            'lag_time': self.lag_time
        }

    @classmethod
    def calculate_location(cls, agent: Aircraft, enemy: Aircraft, lag_time: float = 15) -> tuple[float, float]:
        distance = agent.distance(enemy)
        use_lag_time = max(distance / agent.speed, lag_time)
        target_point = enemy.waypoint.move(-agent.speed * use_lag_time)  # 飞到敌机之前的位置
        return target_point.location

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)
        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return
        target_point = self.calculate_location(lag_time=self.lag_time, agent=self.agent, enemy=enemy)
        yield from go_to_location_updater(self, target_point)
        yield Status.SUCCESS


class AutoPursueNearestEnemy(BTPolicyNode):
    """
    自动判断追逐模式
    根据不同的追逐模式来选择
    - FPolePursueNearestEnemy
    - LeadPursueNearestEnemy
    - LagPursueNearestEnemy
    - PurePursueNearestEnemy
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pursue_mode = ''

    def to_data(self):
        return {
            **super().to_data(),
            'pursue_mode': self.pursue_mode
        }

    @classmethod
    def calculate_pursue_mode(cls, env: Dogfight2dEnv, agent: Aircraft, enemy: Aircraft) -> str:
        missiles = env.battle_area.detect_missiles(agent_name=agent.name)
        if len(missiles) > 0:
            # 目标有处于活动状态的武器
            return 'f-pole'
        distance = agent.distance(enemy)
        positioning = agent.waypoint.calculate_positioning(other=enemy.waypoint)
        distance_threshold = agent.radar_radius * 0.8
        if (distance >= distance_threshold and
                positioning.value not in ['head-to-head',
                                          'head-to-tail'] and enemy.speed >= agent.speed):
            return 'lead'
        elif distance <= distance_threshold and positioning.value not in ['head-to-head', 'head-to-tail']:
            return 'lag'
        else:
            return 'pure'

    @classmethod
    def calculate_location(cls, pursue_mode: str, agent: Aircraft, enemy: Aircraft) -> tuple[float, float]:
        if pursue_mode == 'f-pole':
            return FPolePursueNearestEnemy.calculate_location(agent=agent, enemy=enemy)
        elif pursue_mode == 'lead':
            return LeadPursueNearestEnemy.calculate_location(agent=agent, enemy=enemy)
        elif pursue_mode == 'lag':
            return LagPursueNearestEnemy.calculate_location(agent=agent, enemy=enemy)
        else:
            return PurePursueNearestEnemy.calculate_location(enemy=enemy)

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(
                agent_name=self.agent_name,
                ignore_radar=False)
        if enemy is None:
            self.put_update_message('No nearest enemy')
            yield Status.FAILURE
            return

        self.pursue_mode = self.calculate_pursue_mode(env=self.env, agent=self.agent, enemy=enemy)

        yield from go_to_location_updater(self, location=self.calculate_location(
                pursue_mode=self.pursue_mode,
                enemy=enemy,
                agent=self.agent
        ))


class CheatGoToNearestEnemy(BTPolicyNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.detect_aircraft(agent_name=self.agent_name, ignore_radar=True, only_enemy=True)
        if len(enemy) == 0:
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, location=enemy[0].waypoint.location, keep_time=5)
