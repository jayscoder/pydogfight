from __future__ import annotations
import math
from typing import Tuple, List, Union
import json
import numpy
import numpy as np
from pydogfight.utils.common import *

POINT = Union[Tuple[float, float], List[float]]


class Waypoint:

    def __init__(self, data: np.ndarray | tuple[float, float, float] | list[float] = None):
        if data is None:
            data = [0, 0, 0]
        assert len(data) == 3, "Waypoint must have 3 elements"
        self.data: np.ndarray = np.array(data, dtype=np.float32)
        self.format_coordinates()

    @classmethod
    def build(cls, x: float | np.float32 = 0, y: float | np.float32 = 0, psi: float | np.float32 = 0):
        """
        航迹点
       :param x:
       :param y:
       :param psi: 航向角角度，（0表示正北方向，90度航向角表示正东方向）
        """
        return cls([x, y, psi])

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, value):
        self.data[0] = value

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, value):
        self.data[1] = value

    @property
    def psi(self):
        return self.data[2]

    @psi.setter
    def psi(self, value):
        self.data[2] = value

    @property
    def location(self):
        return self.data[:2]

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", psi: " + str(self.psi)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: 'Waypoint'):
        return np.all(self.data == other.data)

    def to_dict(self):
        return { "x": self.x, "y": self.y, "psi": self.psi }

    def __copy__(self):
        return Waypoint(self.data)

    @property
    def standard_rad(self):
        """
        :return: 与x轴夹角的弧度
        """
        return math.radians(self.standard_angle)

    @property
    def standard_angle(self):
        return heading_to_standard(self.psi)

    def format_coordinates(self, decimal_places: int = 3):
        """
        将 Waypoint 对象的坐标和方向角度格式化为指定小数点后几位
        Args:
            decimal_places: 小数点后保留的位数
        """
        self.data = np.round(self.data, decimal_places)

    def distance(self, other: 'Waypoint' | tuple[float, float] | list[float] | np.ndarray):
        if isinstance(other, np.ndarray):
            return np.linalg.norm(self.data[:2] - other[:2])
        elif isinstance(other, Waypoint):
            return np.linalg.norm(self.data[:2] - other.data[:2])
        else:
            other_data = np.array(other)
            return np.linalg.norm(self.data[:2] - other_data[:2])

    def move_towards(self, target: tuple[float, float] | np.ndarray | list[float], d: float,
                     allow_over: bool = True) -> Waypoint:
        """
        朝着指定目标点飞行一定距离，允许飞过头
        Args:
            target: 目标点
            d: 移动的距离，如果d是负数，表示远离目标点
            allow_over: 是否允许飞过头

        Returns: 移动后的Waypoint
        """
        dist = self.distance(target)
        if dist == 0:
            return self
        ratio = d / dist
        if not allow_over:
            ratio = min(1, ratio)  # 不允许飞过头的情况下，最多飞到目标点的位置

        x = self.x + (target[0] - self.x) * ratio
        y = self.y + (target[1] - self.y) * ratio

        psi = standard_to_heading(math.degrees((math.atan2(y - self.y, x - self.x))))
        return Waypoint.build(x=x, y=y, psi=psi)

    def optimal_move_towards(self, target: tuple[float, float] | np.ndarray, d: float, turn_radius: float):
        # 按照最优轨迹朝着目标点飞行一定距离
        from pydogfight.utils.traj import calc_optimal_path
        next_wpt = calc_optimal_path(
                start=self,
                target=target,
                turn_radius=turn_radius
        ).next_wpt(step=d)
        if next_wpt is None:
            return self.__copy__()
        return next_wpt

    def move(self, d: float, angle: float = 0) -> Waypoint:
        """
        转弯并移动一定距离
        Args:
            d: 距离
            angle: 转弯角度，0代表前进，正值代表向右转，负值代表向左转

        Returns: 移动后的Waypoint

        """
        # 朝着psi的方向移动, psi是航向角，0度指向正北，90度指向正东
        # 将航向角从度转换为弧度
        x_theta = self.standard_rad
        # 计算 x 和 y 方向上的速度分量
        dx = d * math.cos(x_theta)  # 正东方向为正值
        dy = d * math.sin(x_theta)  # 正北方向为正值

        # 更新 obj 的位置
        new_wpt = Waypoint.build(x=self.x + dx, y=self.y + dy, psi=self.psi + angle)
        return new_wpt

    def relative_waypoint(self, other: 'Waypoint') -> Waypoint:
        """以自身为原点的相对航迹点"""
        # dx = other.x - self.x
        # dy = other.y - self.y
        # dpsi = other.psi - self.psi
        return Waypoint(data=other.data - self.data)

    def relative_polar_waypoint(self, other: 'Waypoint') -> PolarWaypoint:
        """
        计算other相对自己的极坐标航迹点
        Args:
            other:

        Returns:

        """
        # 计算距离 r
        r = self.distance(other)

        # 计算角度 theta，结果转换为度
        theta = math.degrees(math.atan2(other.y - self.y, other.x - self.x) - math.radians(self.psi))

        # 保证theta在-180到180度之间
        theta = wrap_angle_to_180(theta)

        # 计算相对朝向 phi，结果转换为度
        phi = other.psi - self.psi

        # 保证phi在-180到180度之间
        phi = wrap_angle_to_180(phi)

        return PolarWaypoint(r, theta, phi)

    def calculate_positioning(self, other: Waypoint, angle_tolerance: float = 30) -> ObjPositioning:
        return ObjPositioning(wpt1=self, wpt2=other, angle_tolerance=angle_tolerance)


class PolarWaypoint:
    def __init__(self, r: float = 0, theta: float = 0, phi: float = 0):
        """
        极坐标航迹点
        :param r: 半径
        :param theta: 角度
        :param phi: 相对朝向 -180到180度之间
        """
        self.data: np.ndarray = np.array([r, theta, phi], dtype=np.float32)

    @property
    def r(self) -> float:
        return self.data[0]

    @r.setter
    def r(self, value: float) -> None:
        self.data[0] = value

    @property
    def theta(self) -> float:
        return self.data[1]

    @theta.setter
    def theta(self, value: float) -> None:
        self.data[1] = value

    @property
    def phi(self) -> float:
        return self.data[2]

    @phi.setter
    def phi(self, value: float) -> None:
        self.data[2] = value

    def __str__(self):
        return "r: " + str(self.r) + ", theta: " + str(self.theta) + ", phi: " + str(self.phi)

    def to_dict(self):
        return { "r": self.r, "theta": self.theta, "phi": self.phi }

    def to_tuple(self):
        return self.r, self.theta, self.phi

    def to_list(self):
        return self.data.tolist()


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


class ObjPositioning:
    """飞机态势关系"""
    HEAD_TO_HEAD = 'head-to-head'
    HEAD_TO_TAIL = 'head-to-tail'
    TAIL_TO_HEAD = 'tail-to-head'
    TAIL_TO_TAIL = 'tail-to-tail'
    ME_FACING_TARGET = 'me-facing-target'
    TARGET_FACING_ME = 'target-facing-me'
    OTHERS = 'others'

    def __init__(self,
                 wpt1: Waypoint,
                 wpt2: Waypoint,
                 angle_tolerance: float = 30):
        self.wpt1 = wpt1
        self.wpt2 = wpt2

        # 2相对于1的极坐标
        rel_2_on_1 = wpt1.relative_polar_waypoint(wpt2)  # r, theta, psi
        # 1相对于2的极坐标
        rel_1_on_2 = wpt2.relative_polar_waypoint(wpt1)  # r, theta, psi
        self.head_1_to_2 = abs(rel_2_on_1.theta) <= angle_tolerance  # 1是否朝向2

        self.head_2_to_1 = abs(rel_1_on_2.theta) <= angle_tolerance  # 2是否朝向1

        self.tail_1_to_2 = abs(rel_2_on_1.theta) >= 180 - angle_tolerance  # 1是否背对2

        self.tail_2_to_1 = abs(rel_1_on_2.theta) >= 180 - angle_tolerance  # 2是否背对1

        if self.head_1_to_2 and self.head_2_to_1:
            self.value = ObjPositioning.HEAD_TO_HEAD
        elif self.head_1_to_2 and self.tail_2_to_1:
            self.value = ObjPositioning.HEAD_TO_TAIL
        elif self.head_1_to_2:
            self.value = ObjPositioning.ME_FACING_TARGET
        elif self.tail_1_to_2 and self.head_2_to_1:
            self.value = ObjPositioning.TAIL_TO_HEAD
        elif self.tail_1_to_2 and self.tail_2_to_1:
            self.value = ObjPositioning.TAIL_TO_TAIL
        elif self.head_2_to_1:
            self.value = ObjPositioning.TARGET_FACING_ME
        else:
            self.value = ObjPositioning.OTHERS

# if __name__ == '__main__':
#     bounding_box = BoundingBox.from_range(x_range=[0, 100], y_range=[0, 100])
#     print(bounding_box.contains([0, 0]))
#     print(bounding_box.contains([-1, -1]))
#     print(bounding_box)
