import math

import gym_dogfight.utils.common as common_utils


class Waypoint:

    def __init__(self, x, y, psi):
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

    @property
    def standard_rad(self):
        """
        :return: 与x轴夹角的弧度
        """
        return math.radians(self.standard_angle)

    @property
    def standard_angle(self):
        return common_utils.heading_to_standard(self.psi)

    def distance(self, other: 'Waypoint'):
        dx = other.x - self.x
        dy = other.y - self
        return (dx * dx + dy * dy) ** 0.5
