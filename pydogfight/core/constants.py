from __future__ import annotations

import numpy as np

TILE_PIXELS = 32
import os


def _hex_color(color: str) -> tuple[int, int, int]:
    # 将十六进制颜色代码转换为RGB值
    r = int(color[0:2], 16)  # 转换红色分量
    g = int(color[2:4], 16)  # 转换绿色分量
    b = int(color[4:6], 16)  # 转换蓝色分量
    return (r, g, b)


# Map of color names to RGB values
COLORS = {
    "red"   : _hex_color('d81e06'),
    "blue"  : _hex_color('1296db'),
    "green" : _hex_color('1afa29'),
    "pink"  : _hex_color('d4237a'),
    "yellow": _hex_color('f4ea2a'),
    "grey"  : _hex_color('8a8a8a'),
    'black' : (0, 0, 0),
    'white' : (255, 255, 255)
}

# # Used to map colors to integers
# COLOR_TO_IDX = { "red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5 }
#
# IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

COLOR_TO_IDX = {
    'white' : 0,
    'red'   : 1,
    'blue'  : 2,
    'green' : 3,
    'pink'  : 4,
    'yellow': 5,
    'grey'  : 6,
    'black' : 7,
}

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen"  : 0,
    "empty"   : 1,
    "aircraft": 2,
    "missile" : 3,
    'home'    : 4,
    'bullseye': 5,  # 战场中心，牛眼
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class DestroyReason:
    """摧毁原因"""
    OUT_OF_GAME_RANGE = "out of game range"  # 超出游戏范围
    # 碰撞导弹
    COLLIDED_WITH_MISSILE = "collided with missile"  # 碰撞导弹
    # 碰撞飞机
    COLLIDED_WITH_AIRCRAFT = "collided with aircraft"  # 碰撞飞机
    # 燃油耗尽
    FUEL_DEPLETION = "fuel depletion"  # 燃油耗尽
    # 基地攻击
    HOME_ATTACK = 'home attack'  # 基地攻击

# class Events:
#     """事件"""
#     # 物体被摧毁
#     # 发射导弹
#     # 飞机碰撞
#     # 导弹命中敌机
#     # 导弹未命中
#     # 回家
