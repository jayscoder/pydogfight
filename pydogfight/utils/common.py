from __future__ import annotations

from typing import Tuple, List
import random
from queue import Queue
import numpy as np
import math


def cal_distance(a: Tuple[float, float] | np.ndarray | list[float],
                 b: Tuple[float, float] | np.ndarray | list[float]) -> float:
    """
    Calculate the distance between two point
    :param a:
    :param b:
    :return:
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# def generate_random_point(rect: ((int, int), (int, int))) -> ((int, int)):

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


def heading_to_standard(hdg):
    # Convert NED heading to standard unit cirlce...degrees only for now (Im lazy)
    thet = wrap_angle_to_360(90 - wrap_angle_to_180(hdg))
    return thet


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
