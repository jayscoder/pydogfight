from __future__ import annotations
import math
from typing import Tuple, List, Union
import numpy as np
import json

POINT = Union[Tuple[float, float], List[float]]


# WAYPOINT = Union[Tuple[float, float, float], List[float]]

class Waypoint:

    def __init__(self, x=0, y=0, psi=0):
        """
        航迹点
        :param x:
        :param y:
        :param psi: 航向角角度，（0表示正北方向，90度航向角表示正东方向）
        """
        self.x = x
        self.y = y
        self.psi = psi

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", psi: " + str(self.psi)

    def to_dict(self):
        return { "x": self.x, "y": self.y, "psi": self.psi }

    def to_tuple(self):
        return self.x, self.y, self.psi

    @property
    def standard_rad(self):
        """
        :return: 与x轴夹角的弧度
        """
        return math.radians(self.standard_angle)

    @property
    def standard_angle(self):
        import pydogfight.utils.common as common_utils
        return common_utils.heading_to_standard(self.psi)

    def distance(self, other: 'Waypoint'):
        dx = other.x - self.x
        dy = other.y - self.y
        return (dx * dx + dy * dy) ** 0.5


class BoundingBox(object):
    left_top: POINT
    size: POINT

    def __init__(self, left_top: POINT, size: POINT):
        self.left_top = left_top
        self.size = size

    def contains(self, point: POINT) -> bool:
        return (self.left_top[0] <= point[0] <= self.left_top[0] + self.size[0]) and (
                self.left_top[1] <= point[1] <= self.left_top[1] + self.size[1])

    def limit(self, point: POINT) -> tuple[float, float]:
        """
        将坐标点 (x, y) 限制在指定的边界范围内。

        Args:
            point (POINT): 坐标值

        Returns:
            tuple: 限制在边界范围内的坐标点
        """
        x, y = point
        x_range = self.x_range
        y_range = self.y_range
        # 将 x 坐标限制在边界范围内
        x = max(x_range[0], min(x_range[1], x))
        # 将 y 坐标限制在边界范围内
        y = max(y_range[0], min(y_range[1], y))
        return x, y

    @classmethod
    def from_center(cls, center: POINT, size: POINT):
        return BoundingBox(
                left_top=(center[0] - size[0] / 2, center[1] - size[1] / 2),
                size=size
        )

    @classmethod
    def from_range(cls, x_range: POINT, y_range: POINT):
        return BoundingBox(
                left_top=(x_range[0], y_range[0]),
                size=(x_range[1] - x_range[0], y_range[1] - y_range[0])
        )

    @property
    def center(self) -> tuple[float, float]:
        return self.left_top[0] + self.size[0] / 2, self.left_top[1] + self.size[1] / 2

    @property
    def right_top(self) -> tuple[float, float]:
        return self.left_top[0] + self.size[0], self.left_top[1]

    @property
    def right_bottom(self) -> tuple[float, float]:
        return self.left_top[0] + self.size[0], self.left_top[1] + self.size[1]

    @property
    def left_bottom(self) -> tuple[float, float]:
        return self.left_top[0], self.left_top[1] + self.size[1]

    @property
    def x_range(self) -> tuple[float, float]:
        return self.left_top[0], self.left_top[0] + self.size[0]

    @property
    def int_x_range(self) -> tuple[int, int]:
        return int(self.left_top[0]), int(self.left_top[0] + self.size[0])

    @property
    def y_range(self) -> tuple[float, float]:
        return self.left_top[1], self.left_top[1] + self.size[1]

    @property
    def int_y_range(self) -> tuple[int, int]:
        return int(self.left_top[1]), int(self.left_top[1] + self.size[1])

    def to_json(self):
        return {
            'left_top'    : list(self.left_top),
            'size'        : list(self.size),
            'center'      : list(self.center),
            'right_top'   : list(self.right_top),
            'right_bottom': list(self.right_bottom),
            'left_bottom' : list(self.left_bottom),
            'x_range'     : list(self.x_range),
            'y_range'     : list(self.y_range)
        }

    def __str__(self):
        return json.dumps(self.to_json(), ensure_ascii=False, indent=4)

    def __repr__(self):
        return json.dumps(self.to_json(), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    bounding_box = BoundingBox.from_range(x_range=[0, 100], y_range=[0, 100])
    print(bounding_box.contains([0, 0]))
    print(bounding_box.contains([-1, -1]))
    print(bounding_box)
