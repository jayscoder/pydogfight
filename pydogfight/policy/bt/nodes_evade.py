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


class Manoeuvre39ToEvadeEnemy(BTPolicyNode):
    @property
    def turn_angle(self) -> int:
        return self.converter.int(self.attrs.get('turn_angle', 90))

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(agent_name=self.agent_name)
        if enemy is None:
            yield Status.FAILURE
            return

        target = generate_39_manoeuvre(self.agent.waypoint, enemy_wpt=enemy.waypoint,
                                       dis=self.agent.turn_radius * 3, turn_angle=self.turn_angle)

        yield from go_to_location_updater(self, (target[0], target[1]))


class Manoeuvre39ToEvadeMissile(BTPolicyNode):

    @property
    def turn_angle(self) -> int:
        return self.converter.int(self.attrs.get('turn_angle', 90))

    def updater(self) -> typing.Iterator[Status]:
        from pydogfight.core.world_obj import Missile
        missiles: list[Missile] = self.env.battle_area.detect_missiles(agent_name=self.agent_name)
        if len(missiles) == 0:
            yield Status.FAILURE
            return

        enemy_waypoints = [missile.waypoint for missile in missiles]

        target = generate_weighted_manoeuvre(self.agent.waypoint, enemy_wpts=enemy_waypoints,
                                             dis=self.agent.turn_radius * 3, turn_angle=self.turn_angle)
        yield from go_to_location_updater(self, (target[0], target[1]))


def generate_weighted_manoeuvre(self_wpt: Waypoint, enemy_wpts: list[Waypoint], dis: float = 100, turn_angle: int = 90):
    weighted_x, weighted_y = 0, 0
    total_weight = 0

    for waypoint in enemy_wpts:
        # Calculate distance between the agent and the missile
        distance = math.sqrt((waypoint.x - self_wpt.x) ** 2 + (waypoint.y - self_wpt.y) ** 2)
        weight = 1 / (distance + 1e-6)  # Weight inversely proportional to distance, avoid division by zero
        total_weight += weight

        angle = math.atan2(self_wpt.y - waypoint.y, self_wpt.x - waypoint.x) * 180 / math.pi

        new_x = self_wpt.x + dis * math.cos(math.radians(angle + turn_angle))
        new_y = self_wpt.y + dis * math.sin(math.radians(angle + turn_angle))

        weighted_x += new_x * weight
        weighted_y += new_y * weight

    # Calculate the final weighted coordinates
    if total_weight > 0:
        weighted_x /= total_weight
        weighted_y /= total_weight

    new_psi = self_wpt.psi  # Keep the same heading as the original target

    return weighted_x, weighted_y, new_psi


def generate_39_manoeuvre(self_wpt: Waypoint, enemy_wpt: Waypoint, dis: float = 100, turn_angle: int = 90):
    # Calculate distance and angle between the two points
    # distance = math.sqrt((enemy_wpt.x - self_wpt.x) ** 2 + (enemy_wpt.y - self_wpt.y) ** 2)
    angle = math.atan2(self_wpt.y - enemy_wpt.y, self_wpt.x - enemy_wpt.x) * 180 / math.pi

    # 39 Manoeuvre: Move perpendicular to the line connecting the missile and target
    # Choose a distance (you can adjust this based on your specific requirements)
    manoeuvre_distance = dis  # Example distance

    # Calculate new target point coordinates
    new_x = self_wpt.x + manoeuvre_distance * math.cos(math.radians(angle + turn_angle))
    new_y = self_wpt.y + manoeuvre_distance * math.sin(math.radians(angle + turn_angle))
    new_psi = self_wpt.psi  # Keep the same heading as the original target

    return new_x, new_y, new_psi
