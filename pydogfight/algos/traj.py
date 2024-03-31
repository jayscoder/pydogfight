from __future__ import annotations

import time

from pydogfight.core.models import Waypoint
import math
import numpy as np
from pydogfight.algos.geometry import *
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

    def next_wpt(self, step: float) -> Waypoint | None:
        """
        生成下一步的航迹
        :param step: 步长
        :return:
        """
        if self.length == float('inf'):
            return None

        turn_points_count = int(np.floor(self.turn_length / step))
        direct_points_count = int(np.floor(self.direct_length / step))

        # 先生成拐弯的点
        if turn_points_count > 0:
            init_theta = math.atan2(self.start.y - self.turn_center[1], self.start.x - self.turn_center[0])
            curr_turn_rad = np.deg2rad(self.turn_angle) / turn_points_count
            x = self.turn_center[0] + (self.turn_radius * np.cos(init_theta + curr_turn_rad))
            y = self.turn_center[1] + (self.turn_radius * np.sin(init_theta + curr_turn_rad))
            psi = self.start.psi - np.rad2deg(curr_turn_rad)
            return Waypoint(x=x, y=y, psi=psi)

        # 生成直线点
        if direct_points_count > 0:
            curr_direct_length = self.direct_length / direct_points_count
            target_rad = np.deg2rad(90 - self.target.psi)
            x = self.turn_point[0] + curr_direct_length * np.cos(target_rad)
            y = self.turn_point[1] + curr_direct_length * np.sin(target_rad)
            psi = self.target.psi
            return Waypoint(x=x, y=y, psi=psi)

        return None

    def generate_traj(self, step: float) -> np.ndarray | None:
        """
        生成最优航迹
        :param step: 每一步的长度
        :return:
        """
        if self.length == float('inf'):
            return

        turn_points_count = int(np.floor(self.turn_length / step))
        direct_points_count = int(np.floor(self.direct_length / step))
        traj = []

        # 先生成拐弯的点
        if turn_points_count > 0:
            init_theta = math.atan2(self.start.y - self.turn_center[1], self.start.x - self.turn_center[0])
            curr_turn_rad = np.linspace(0, np.deg2rad(self.turn_angle), turn_points_count)
            x = self.turn_center[0] + (self.turn_radius * np.cos(init_theta + curr_turn_rad))
            y = self.turn_center[1] + (self.turn_radius * np.sin(init_theta + curr_turn_rad))
            psi = self.start.psi - np.rad2deg(curr_turn_rad)
            turn_traj = np.stack([x, y, psi], axis=1)
            traj.append(turn_traj)

        # 生成直线点
        if direct_points_count > 0:
            curr_direct_length = np.linspace(0, self.direct_length, direct_points_count)
            target_rad = np.deg2rad(90 - self.target.psi)
            x = self.turn_point[0] + curr_direct_length * np.cos(target_rad)
            y = self.turn_point[1] + curr_direct_length * np.sin(target_rad)
            psi = self.target.psi * np.ones_like(curr_direct_length)
            direct_traj = np.stack([x, y, psi], axis=1)
            traj.append(direct_traj)

        if len(traj) == 0:
            return None

        return np.concatenate(traj, axis=0)


def calc_optimal_path(
        start: Waypoint | tuple[float, float, float],
        target: Waypoint | tuple[float, float],
        turn_radius: float) -> OptimalPathParam:
    """
    计算最短航迹
    psi: 航向角（角度），0代表正北，90代表正东
    :param start: 起点
    :param target: 目标点
    :param turn_radius: 转弯半径
    :return:
    """
    if not isinstance(start, Waypoint):
        start = Waypoint(x=start[0], y=start[1], psi=start[2])
    if not isinstance(target, Waypoint):
        target = Waypoint(x=target[0], y=target[1], psi=0)

    # 将角度从度转换为弧度，且是与x轴正方向的夹角
    param = OptimalPathParam(start=start, target=target, turn_radius=turn_radius)
    param.turn_radius = turn_radius

    start_target_distance = np.sqrt((target.x - start.x) ** 2 + (target.y - start.y) ** 2)

    if start_target_distance == 0:
        param.length = 0
        param.direct_length = 0
        param.target.psi = param.start.psi
        param.turn_point = (param.start.x, param.start.y)
        return param

    start_to_target_theta = math.atan2(target.y - start.y, target.x - start.x)  # 终点到起点连线与x轴的夹角

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
                x1=target.x,
                y1=target.y)
        for point in tangent_points:
            center_to_point_vector = (point[0] - circle_center[0], point[1] - circle_center[1])  # 圆心到拐点的向量
            point_to_target_vector = (target.x - point[0], target.y - point[1])  # 拐点到目标点向量
            point_to_target_vector_rotate_sign = sign(
                    cross(center_to_point_vector, point_to_target_vector))  # 拐点旋转方向（正代表逆时针，负代表顺时针）
            if start_vector_rotate_sign != point_to_target_vector_rotate_sign:
                # 起始的旋转方向必须和最终的旋转方向一致
                continue

            direct_length = math.sqrt((target.x - point[0]) ** 2 + (target.y - point[1]) ** 2)
            point_to_target_theta = math.atan2(target.y - point[1], target.x - point[0])

            turn_rad = clockwise_rotation_rad(start_vector_rotate_sign, center_to_start_vector, center_to_point_vector)
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


def test_bench():
    N = 10
    start_time = time.time()
    for i in range(N):
        start = Waypoint(16045.390142119586, 14000.726933966973, 117)
        target = (-12187, 7000)
        param = calc_optimal_path(start, target, 5000)
        param.generate_traj(step=1)
    end_time = time.time()
    cost_time = end_time - start_time
    print("One Time: " + str(cost_time / N))  # v1: 0.065 v2: 0.032 v3: 0.0008
    print("Total time: " + str(cost_time))


def test_main():
    import matplotlib.pyplot as plt
    # User's waypoints: [x, y, heading (degrees)]

    start = Waypoint(0, 0, 90)
    target = (100, 10)
    param = calc_optimal_path(start, target, 10)
    if param.length != float('inf'):
        path = param.generate_traj(step=1)
        print(param)
        print('path shape', path.shape)
        # Plot the results

        if param.turn_center is not None:
            fig, ax = plt.subplots()
            circle = plt.Circle(param.turn_center, param.turn_radius, fill=False)
            ax.add_patch(circle)
            plt.plot(param.turn_center[0], param.turn_center[1], 'kx')

        if param.turn_point is not None:
            plt.plot(param.turn_point[0], param.turn_point[1], 'kx')
            plt.quiver(param.turn_point[0], param.turn_point[1], np.cos(param.target.standard_rad),
                       np.sin(param.target.standard_rad),
                       scale=10, color='green',
                       label="Turn Direction")

        plt.plot(start.x, start.y, 'kx')
        plt.plot(target[0], target[1], 'kx')

        plt.quiver(start.x, start.y, np.cos(start.standard_rad), np.sin(start.standard_rad), scale=10, color='green',
                   label="Initial Direction")

        plt.plot(path[:, 0], path[:, 1], 'b-')
        plt.grid(True)
        plt.axis("equal")
        plt.title('Optimal Trajectory Generation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


if __name__ == '__main__':
    test_main()
