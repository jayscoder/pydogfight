from __future__ import annotations

from typing import Tuple, List
import random
from queue import Queue
import numpy as np
import math
import numpy as np


def cal_distance(a: Tuple[float, float] | np.ndarray | list[float],
                 b: Tuple[float, float] | np.ndarray | list[float]) -> float:
    """
    Calculate the distance between two point
    :param a:
    :param b:
    :return:
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def wrap_angle_to_360(angle):
    """
    将角度处理到[0, 360] 以内
    """
    posIn = angle > 0
    angle = angle % 360
    if angle == 0 and posIn:
        angle = 360
    return angle


def wrap_angle_to_180(angle):
    """
    将角度放缩到[-180, 180]之内
    """
    if (angle < -180) or (180 < angle):
        angle = wrap_angle_to_360(angle + 180) - 180
    return angle


def heading_to_standard(psi):
    """
    航向角是指无人机当前运动方向与地面参考方向之间的夹角，通常用正北方向为参考方向。
    将航向角（NED，以正北为0度，正东为90度）转换为标准单位圆角度
    Args:
        psi: 航向角 90度表示正东，0度表示正北
    Returns: 以标准单位圆角度表示的航向角，范围[0, 360)

    """
    # Convert NED heading to standard unit cirlce...degrees only for now (Im lazy)
    return wrap_angle_to_360(90 - wrap_angle_to_180(psi)) % 360


def standard_to_heading(angle):
    """
    将标准单位圆角度（0-360度）转换为航向角（NED，以正北为0度，正东为90度）。
    Args:
        angle: 标准单位圆角度

    Returns: 航向角，90度表示正东，0度表示正北，范围是(-180, 180]

    """
    angle = wrap_angle_to_180(90 - wrap_angle_to_360(angle))
    if angle == -180:
        angle = 180
    return angle


def generate_random_point(top: Tuple[float, float], size: Tuple[float, float]) -> Tuple[float, float]:
    x_range = (top[0], top[0] + size[0])
    y_range = (top[1], top[1] + size[1])

    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])

    return x, y


def read_queue_without_destroying(q: Queue):
    # 创建一个空列表来存储队列中的元素
    temp_list = []

    # 遍历队列，复制元素到列表和临时队列
    while not q.empty():
        item = q.get_nowait()
        temp_list.append(item)

    # 将元素重新放入原始队列，保持原始状态不变
    for item in temp_list:
        q.put_nowait(item)

    return temp_list


def will_collide(a1: np.ndarray, a2: np.ndarray, ra: float, b1: np.ndarray, b2: np.ndarray, rb: float):
    """
    计算两个在时间段内移动的物体是否碰撞。

    参数:
    a1, a2 : 物体A在时间段开始和结束时的位置 (numpy数组)
    ra : 物体A的碰撞半径
    b1, b2 : 物体B在时间段开始和结束时的位置 (numpy数组)
    rb : 物体B的碰撞半径

    返回:
    如果物体在时间段内碰撞，返回True，否则返回False。
    """
    # 计算两个物体中心间的最小距离是否小于等于碰撞半径之和
    # 速度向量
    va = a2 - a1
    vb = b2 - b1

    # 碰撞半径之和
    collision_distance = ra + rb

    # 将问题简化为点与圆的最近距离问题
    # 设t为时间参数，t从0到1，我们找到距离函数d(t)的最小值
    # d(t)^2 = |(a1 + t * va) - (b1 + t * vb)|^2
    #  = |(a1 - b1) + t * (va - vb)|^2
    # 我们需要计算的是，这个距离的平方是否在任意时刻小于等于 collision_distance^2

    # 相对速度
    v_rel = va - vb

    # 初始距离向量
    initial_offset = a1 - b1

    # 利用求导数的方法找到最小距离
    # d(t)^2的导数为：2 * (initial_offset + t * v_rel) * v_rel = 0
    # 解得 t = - (initial_offset * v_rel) / (v_rel * v_rel)

    if np.dot(v_rel, v_rel) == 0:
        # 如果相对速度为零，说明两物体保持相对静止
        # 只需要比较初始距离
        min_distance_squared = np.dot(initial_offset, initial_offset)
    else:
        t = -np.dot(initial_offset, v_rel) / np.dot(v_rel, v_rel)
        # 确保t在[0,1]之间
        t = max(0, min(1, t))
        # 计算t时刻的位置
        closest_point = initial_offset + t * v_rel
        min_distance_squared = np.dot(closest_point, closest_point)

    return min_distance_squared <= collision_distance ** 2
