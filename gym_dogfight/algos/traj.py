from __future__ import annotations

from gym_dogfight.core.models import Waypoint
import math
import numpy as np
from gym_dogfight.algos.geometry import *
from enum import Enum


class OptimalPathParam:
    start: Waypoint
    target: Waypoint

    turn_radius: float

    length = float('inf')  # 航迹长度
    turn_angle = 0
    turn_length = 0
    turn_point = None  # 拐点
    turn_center = None  # 拐弯的圆心
    direct_length = 0

    def __init__(self, start: Waypoint, target: Waypoint, turn_radius: float):
        self.start = start
        self.target = target
        self.turn_radius = turn_radius
        self.turn_point = (start.x, start.y)

    def __str__(self):
        return '\n'.join([
            f'start: {self.start}',
            f'target: {self.target}',
            f'turn_radius: {self.turn_radius}',
            f'turn_angle: {self.turn_angle}',
            f'turn_length: {self.turn_length}',
            f'turn_point: {self.turn_point}',
            f'turn_center: {self.turn_center}',
            f'direct_length: {self.direct_length}',
        ])

    def generate_traj(self, step: float):
        """
        生成最优航迹
        :param step: 每一步的长度
        :return:
        """
        if self.length == float('inf'):
            return []

        turn_points_count = math.floor(self.turn_length / step)
        direct_points_count = math.floor(self.direct_length / step)
        path = -1 * np.ones((turn_points_count + direct_points_count, 3))

        # 先生成拐弯的点
        if turn_points_count > 0:
            turn_rad_step = math.radians(self.turn_angle) / turn_points_count
            curr_turn_rad = 0
            init_theta = math.atan2(self.start.y - self.turn_center[1], self.start.x - self.turn_center[0])

            for i in range(turn_points_count):
                curr_turn_rad += turn_rad_step
                path[i][0] = self.turn_center[0] + (self.turn_radius * math.cos(init_theta + curr_turn_rad))
                path[i][1] = self.turn_center[1] + (self.turn_radius * math.sin(init_theta + curr_turn_rad))
                path[i][2] = self.start.psi - math.degrees(curr_turn_rad)

        # 生成直线点
        curr_direct_length = 0
        target_rad = math.radians(90 - self.target.psi)
        for i in range(turn_points_count, turn_points_count + direct_points_count):
            curr_direct_length += step
            path[i][0] = self.turn_point[0] + curr_direct_length * math.cos(target_rad)
            path[i][1] = self.turn_point[1] + curr_direct_length * math.sin(target_rad)
            path[i][2] = self.target.psi

        return path


def calc_optimal_path(start: Waypoint, target: tuple[float, float], turn_radius: float) -> OptimalPathParam:
    """
    计算最短航迹
    psi: 航向角（角度），0代表正北，90代表正东
    :param start: 起点
    :param target: 目标点
    :param turn_radius: 转弯半径
    :return:
    """

    # 将角度从度转换为弧度，且是与x轴正方向的夹角
    param = OptimalPathParam(start=start, target=Waypoint(target[0], target[1], 0), turn_radius=turn_radius)
    param.turn_radius = turn_radius

    start_target_distance = math.sqrt((target[0] - start.x) ** 2 + (target[1] - start.y) ** 2)

    if start_target_distance == 0:
        param.length = 0
        param.direct_length = 0
        param.target.psi = param.start.psi
        param.turn_point = (param.start.x, param.start.y)
        return param

    start_to_target_theta = math.atan2(target[1] - start.y, target[0] - start.x)  # 终点到起点连线与x轴的夹角

    if start_to_target_theta == start.standard_rad:
        # 直接沿着直线飞行
        param.length = start_target_distance
        param.direct_length = start_target_distance
        param.target.psi = param.start.psi
        param.turn_point = (param.start.x, param.start.y)
        return param

    # 先拐弯，再直线飞行
    # 计算与初始航线相切圆的圆心
    circle_centers = calculate_tangent_circle_centers(start.x, start.y, 90 - start.psi, turn_radius)

    start_vector = (math.cos(start.standard_rad), math.sin(start.standard_rad))  # 初始方向向量

    # 计算相切圆与目标点的切点
    for circle_center in circle_centers:
        # if not are_points_on_same_side_of_line(circle_center, target, start_direct_line_coef):
        #     continue
        center_to_start_vector = (start.x - circle_center[0], start.y - circle_center[1])  # 圆心到初始点的向量
        start_vector_rotate_sign = sign(cross(center_to_start_vector, start_vector))  # 初始向量旋转方向（正代表逆时针，负代表顺时针）

        tangent_points = calculate_circle_tangent_points(
                x0=circle_center[0],
                y0=circle_center[1],
                r=turn_radius,
                x1=target[0],
                y1=target[1])
        for point in tangent_points:
            center_to_point_vector = (point[0] - circle_center[0], point[1] - circle_center[1])  # 圆心到拐点的向量
            point_to_target_vector = (target[0] - point[0], target[1] - point[1])  # 拐点到目标点向量
            point_to_target_vector_rotate_sign = sign(
                    cross(center_to_point_vector, point_to_target_vector))  # 拐点旋转方向（正代表逆时针，负代表顺时针）
            if start_vector_rotate_sign != point_to_target_vector_rotate_sign:
                # 起始的旋转方向必须和最终的旋转方向一致
                continue

            direct_length = math.sqrt((target[0] - point[0]) ** 2 + (target[1] - point[1]) ** 2)
            point_to_target_theta = math.atan2(target[1] - point[1], target[0] - point[0])

            turn_rad = clockwise_rotation_rad(start_vector_rotate_sign, center_to_start_vector, center_to_point_vector)
            # if start_vector_rotate_sign > 0 > turn_rad:
            #     # 起始方向是逆时针旋转
            #     turn_rad = math.pi * 2 - abs(turn_rad)
            # elif start_vector_rotate_sign < 0 < turn_rad:
            #     # 起始方向是顺时针旋转
            #     turn_rad = math.pi * 2 - turn_rad

            turn_length = abs(math.pi * turn_rad * turn_radius)
            total_length = direct_length + turn_length
            if total_length < param.length:
                param.length = total_length
                param.direct_length = direct_length
                param.turn_angle = math.degrees(turn_rad)
                param.turn_length = turn_length
                param.turn_point = point
                param.turn_center = circle_center
                param.target.psi = 90 - math.degrees(point_to_target_theta)

    return param


def main():
    import matplotlib.pyplot as plt
    # User's waypoints: [x, y, heading (degrees)]

    start = Waypoint(16045.390142119586, 14000.726933966973, 117)
    target = (-12187, 7000)
    param = calc_optimal_path(start, target, 5000)
    if param.length != float('inf'):
        path = param.generate_traj(step=100)
        print(param)
        print(path.shape[0])
        # Plot the results

        circle = plt.Circle(param.turn_center, param.turn_radius, fill=False)
        fig, ax = plt.subplots()
        ax.add_patch(circle)

        plt.plot(start.x, start.y, 'kx')
        plt.plot(target[0], target[1], 'kx')
        plt.plot(param.turn_center[0], param.turn_center[1], 'kx')
        if param.target is not None:
            plt.plot(param.turn_point[0], param.turn_point[1], 'kx')

        plt.quiver(start.x, start.y, np.cos(start.standard_rad), np.sin(start.standard_rad), scale=10, color='green',
                   label="Initial Direction")

        plt.quiver(param.turn_point[0], param.turn_point[1], np.cos(param.target.standard_rad),
                   np.sin(param.target.standard_rad),
                   scale=10, color='green',
                   label="Turn Direction")

        # for point in path:
        #     plt.quiver(point[0], point[1], np.cos(math.radians(90 - point[2])),
        #                np.sin(math.radians(90 - point[2])),
        #                scale=10, color='green',
        #                label="")

        plt.plot(path[:, 0], path[:, 1], 'b-')
        plt.grid(True)
        plt.axis("equal")
        plt.title('Optimal Trajectory Generation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


if __name__ == '__main__':
    main()
