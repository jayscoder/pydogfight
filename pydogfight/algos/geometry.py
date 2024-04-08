from __future__ import annotations

from typing import Tuple

import numpy as np

import math
import time

import gymnasium as gym
import pydogfight


# 计算直线的切圆圆心位置
def calculate_tangent_circle_centers(x0: float, y0: float, angle: float, radius: float):
    """
    计算与给定直线相切的圆的圆心坐标。

    :param x0: 直线上的点的x坐标
    :param y0: 直线上的点的y坐标
    :param angle: 直线与x轴的夹角（以度为单位）
    :param radius: 圆的半径
    :return: 两个可能的圆心坐标
    """
    # 将角度转换为弧度
    angle_rad = math.radians(angle)

    # 计算法线的角度
    angle_normal_1 = angle_rad + math.pi / 2
    angle_normal_2 = angle_rad - math.pi / 2

    # 计算两个可能的圆心坐标
    circle_center_1 = (x0 + radius * math.cos(angle_normal_1), y0 + radius * math.sin(angle_normal_1))
    circle_center_2 = (x0 + radius * math.cos(angle_normal_2), y0 + radius * math.sin(angle_normal_2))

    return [circle_center_1, circle_center_2]


def calculate_circle_tangent_points(x0, y0, r, x1, y1):
    """
    Calculate the tangent points on a circle from an external point.

    :param x0: Circle center x-coordinate
    :param y0: Circle center y-coordinate
    :param r: Circle radius
    :param x1: External point x-coordinate
    :param y1: External point y-coordinate
    :return: A list of tuples, each tuple is a coordinate of a tangent point on the circle
    """
    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if d < r:
        return []  # No tangent points if the point is inside the circle

    if d == r:
        return [(x1, y1)]

    alpha = math.atan2(y1 - y0, x1 - x0)
    theta = math.asin(r / d)

    angle1 = math.pi / 2 - theta + alpha
    angle2 = math.pi / 2 - theta - alpha

    tangent_points = [
        (x0 + r * math.cos(angle1), y0 + r * math.sin(angle1)),
        (x0 + r * math.cos(angle2), y0 - r * math.sin(angle2))
    ]

    return tangent_points


def are_points_on_same_side_of_line(p1, p2, line_coefficients):
    """
    Determine if two points are on the same side of a given line.

    :param p1: The first point as a tuple (x, y)
    :param p2: The second point as a tuple (x, y)
    :param line_coefficients: The coefficients of the line equation ax + by + c = 0 as a tuple (a, b, c)
    :return: True if both points are on the same side of the line, False otherwise
    """
    a, b, c = line_coefficients
    x1, y1 = p1
    x2, y2 = p2

    d1 = a * x1 + b * y1 + c
    d2 = a * x2 + b * y2 + c

    # Both points are on the same side of the line if d1 and d2 have the same sign
    return d1 * d2 > 0


def extract_line_coefficients(x0, y0, theta) -> Tuple[float, float, float]:
    """
    Extract the coefficients of the line from point and theta
    :param x0:
    :param y0:
    :param theta:
    :return: The coefficients of the line equation ax + by + c = 0 as a tuple (a, b, c)
    """
    if theta == 90:
        return x0, 0, 0
    k = math.tan(theta)
    a = k
    b = -1
    c = -k * x0 + y0
    return a, b, c


def sign(x: float) -> float:
    if x >= 0:
        return 1
    else:
        return -1


def cross(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    return v1[0] * v2[1] - v1[1] * v2[0]


def wrap_angle_to_360(angle):
    posIn = angle > 0
    angle = angle % 360
    if angle == 0 and posIn:
        angle = 360
    return angle


def wrap_angle_to_180(angle):
    q = (angle < -180) or (180 < angle)
    if (q):
        angle = wrap_angle_to_360(angle + 180) - 180
    return angle


# def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
#     return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def rad_between_vectors(v1, v2):
    # 计算向量的夹角
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    # 计算点乘
    dot_product = np.dot(v1, v2)

    # 计算夹角的余弦值
    cos_angle = dot_product / (v1_norm * v2_norm)

    # 使用 arccos 函数计算夹角（弧度制）
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return angle_rad


def clockwise_rotation_rad(clockwise_type, v1, v2):
    """
    # 计算两个向量之间的夹角
    :param clockwise_type: 1代表逆时针，-1代表顺时针
    :param v1:
    :param v2:
    :return:
    """
    rad = rad_between_vectors(v1, v2)
    assert math.pi > rad >= 0
    # 计算旋转的方向
    if sign(cross(v1, v2)) != clockwise_type:
        rad = math.pi * 2 - rad

    if clockwise_type == 1:
        return rad
    else:
        return -rad
