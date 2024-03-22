from __future__ import annotations

from gym_dogfight.core.models import Waypoint
from typing import List, Callable, Tuple
import numpy as np


def predict_intercept_point(
        target: Waypoint,
        target_speed: float,
        self_speed: float,
        calc_optimal_dis: Callable[[Tuple[float, float]], float],
        precision: float = 0.5
) -> Tuple[
         float, float] | None:
    """
    预测拦截目标点
    :param target:
    :param target_speed: 敌方的速度 m/s
    :param self_speed: 我方的速度 m/s
    :param calc_optimal_dis: 计算我方飞到目标点的距离函数（黑盒函数）
    :param precision: 计算精度，拦截误差小于precision即可认为拦截成功，单位m
    :return: Tuple[float, float]
    """

    def _calc(d: float):
        # 最多尝试100次
        hit_point = (target.x + d * np.cos(target.standard_rad), target.y + d * np.sin(target.standard_rad))
        mt = calc_optimal_dis(hit_point) / self_speed  # 导弹飞到目标点需要多久
        et = d / target_speed
        return mt, et, hit_point

    d = 0  # 预测敌机飞行距离
    for i in range(100):
        # 最多尝试100次
        mt, et, hit_point = _calc(d)
        # print(f'i={i}, mt={mt}, et={et}, d={d}')
        if mt == float('inf') or et == float('inf'):
            return None
        diff_t = abs(mt - et)
        if diff_t * (target_speed + self_speed) < precision:
            return hit_point
        if mt > et:
            # 敌人先到目标点，d需要增大
            d += diff_t * target_speed
        elif mt < et:
            # 敌人后到目标点，d需要减小
            d -= diff_t * target_speed

    return None



# def predict_missile_hit_prob(self, source: Aircraft, target: Aircraft):
#     """
#     预测导弹命中目标概率
#     根据距离来判断敌方被摧毁的概率，距离越远，被摧毁的概率越低（基于MISSILE_MAX_THREAT_DISTANCE和MISSILE_NO_ESCAPE_DISTANCE）
#     :param source: 发射导弹方
#     :param target: 被导弹攻击方
#     :return:
#     """
#     # 计算导弹发射轨迹
#     from gym_dogfight.algos.traj import calc_optimal_path
#     hit_point = source.predict_missile_intercept_point(enemy=target)
#
#     if hit_point is None:
#         return 0
#
#     param = calc_optimal_path(
#             start=source.waypoint,
#             target=(target.x, target.y),
#             turn_radius=self.missile_min_turn_radius
#     )
#
#     # 如果距离小于等于不可躲避距离，目标必定被摧毁
#     if param.length <= self.missile_no_escape_distance:
#         return 1
#
#     # 如果距离超出最大威胁距离，目标不会被摧毁
#     if param.length > self.missile_max_threat_distance:
#         return 0
#
#     # 在不可躲避距离和最大威胁距离之间，摧毁的概率随距离增加而减少
#     hit_prob = (self.missile_max_threat_distance - param.length) / (
#             self.missile_max_threat_distance - self.missile_no_escape_distance)
#
#     return hit_prob


def _main():
    import matplotlib.pyplot as plt
    from gym_dogfight.algos.traj import calc_optimal_path

    enemy = Waypoint(100, 100, 90)
    enemy_speed = 1
    missile_speed = 10
    missile = Waypoint(0, 0, 0)
    missile_turn_radius = 10
    enemy_turn_radius = 5

    def calc_missile_hit_dis(p: Tuple[float, float]):
        param = calc_optimal_path(missile, p, missile_turn_radius)
        if param is None:
            return float('inf')
        return param.length

    hit_point = predict_intercept_point(enemy, enemy_speed, missile_speed, calc_missile_hit_dis)

    plt.quiver(enemy.x, enemy.y, np.cos(enemy.standard_rad), np.sin(enemy.standard_rad), scale=10, color='blue',
               label="Enemy")
    plt.quiver(missile.x, missile.y, np.cos(missile.standard_rad), np.sin(missile.standard_rad), scale=10, color='red',
               label="Missile")

    if hit_point is not None:
        plt.plot(hit_point[0], hit_point[1], 'kx')

        hit_param = calc_optimal_path(missile, hit_point, missile_turn_radius)
        hit_path = np.array(list(hit_param.generate_traj(1)))

        plt.plot(hit_path[:, 0], hit_path[:, 1], 'b-')

        enemy_param = calc_optimal_path(enemy, hit_point, enemy_turn_radius)
        enemy_path = np.array(list(enemy_param.generate_traj(1)))
        plt.plot(enemy_path[:, 0], enemy_path[:, 1], 'b-')

        print('hit_path', hit_path.shape)
        print('enemy_path', enemy_path.shape)
    else:
        print('没有发现HitPoint')

    plt.grid(True)
    plt.axis("equal")
    plt.title('Optimal Trajectory Generation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    _main()
