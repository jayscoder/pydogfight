from __future__ import annotations

from pydogfight.utils.traj import calc_optimal_path
from pydogfight.policy.bt.common import *


class EvadeMissile(BTPolicyNode):
    """
    行为节点: 规避导弹
    - SUCCESS: 规避成功
    - FAILURE: 规避失败（可能是不存在来袭导弹、无法规避导弹）
    """

    def __init__(self, test_move_angle_sep: int | str = 45, **kwargs):
        super().__init__(**kwargs)
        self.test_move_angle_sep = test_move_angle_sep

    def updater(self) -> typing.Iterator[Status]:
        # 获取导弹
        missiles = self.env.battle_area.detect_missiles(agent_name=self.agent_name, ignore_radar=False, only_enemy=True)
        if len(missiles) == 0:
            yield Status.FAILURE
            return

        test_waypoints = self.agent.generate_test_moves(
                in_safe_area=True,
                angle_sep=self.converter.int(self.test_move_angle_sep))

        # 从周围N个点中寻找一个能够让导弹飞行时间最长且自己飞行时间最短的点来飞 （导弹飞行时间 - 自己飞行时间）最大
        max_diff_time = -float('inf')
        go_to_location = None
        for waypoint in test_waypoints:
            diff_time = 0
            for mis in missiles[:1]:  # 只规避最近的导弹
                under_hit_point = mis.predict_intercept_point(target_wpt=waypoint, target_speed=self.agent.speed)
                if under_hit_point is None:
                    continue
                if under_hit_point.time == float('inf'):
                    under_hit_point.time = 10000

                optimal_path = calc_optimal_path(
                        start=self.agent.waypoint,
                        target=waypoint.location,
                        turn_radius=self.agent.turn_radius,
                )
                if optimal_path.length == float('inf'):
                    continue

                diff_time += under_hit_point.time - optimal_path.length / self.agent.speed

            if diff_time > max_diff_time:
                max_diff_time = diff_time
                go_to_location = waypoint.location

        if go_to_location is None:
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, go_to_location)


class Manoeuvre39ToEvadeMissile(BTPolicyNode):

    def updater(self) -> typing.Iterator[Status]:
        from pydogfight.core.world_obj import Missile
        missiles: list[Missile] = self.env.battle_area.detect_missiles(agent_name=self.agent_name)
        if len(missiles) == 0:
            yield Status.FAILURE
            return

        target = generate_39_manoeuvre(self.agent.waypoint, enemy_wpt=missiles[0].waypoint,
                                       dis=self.agent.turn_radius * 3)

        yield from go_to_location_updater(self, (target[0], target[1]))


class Manoeuvre39ToEvadeEnemy(BTPolicyNode):

    def updater(self) -> typing.Iterator[Status]:
        from pydogfight.core.world_obj import Missile
        enemy = self.env.battle_area.find_nearest_enemy(agent_name=self.agent_name)
        if enemy is None:
            yield Status.FAILURE
            return

        target = generate_39_manoeuvre(self.agent.waypoint, enemy_wpt=enemy.waypoint,
                                       dis=self.agent.turn_radius * 3)

        yield from go_to_location_updater(self, (target[0], target[1]))


def generate_39_manoeuvre(self_wpt: Waypoint, enemy_wpt: Waypoint, dis: float = 100):
    # Calculate distance and angle between the two points
    distance = math.sqrt((enemy_wpt.x - self_wpt.x) ** 2 + (enemy_wpt.y - self_wpt.y) ** 2)
    angle = math.atan2(self_wpt.y - enemy_wpt.y, self_wpt.x - enemy_wpt.x) * 180 / math.pi

    # 39 Manoeuvre: Move perpendicular to the line connecting the missile and target
    # Choose a distance (you can adjust this based on your specific requirements)
    manoeuvre_distance = dis  # Example distance

    # Calculate new target point coordinates
    new_x = self_wpt.x + manoeuvre_distance * math.cos(math.radians(angle + 90))
    new_y = self_wpt.y + manoeuvre_distance * math.sin(math.radians(angle + 90))
    new_psi = self_wpt.psi  # Keep the same heading as the original target

    return new_x, new_y, new_psi



