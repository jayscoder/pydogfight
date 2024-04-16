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
