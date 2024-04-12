import unittest

import numpy as np

from pydogfight.utils.common import *
import random


class TestCommon(unittest.TestCase):

    def test_wrap_angle_to_360(self):
        self.assertEqual(180, wrap_angle_to_360(180))
        self.assertEqual(180, wrap_angle_to_360(180 + 360))
        self.assertEqual(360, wrap_angle_to_360(360 + 360))
        self.assertEqual(180, wrap_angle_to_360(180 - 360))
        self.assertEqual(90, wrap_angle_to_360(90 - 360))
        self.assertEqual(90, wrap_angle_to_360(90 - 360 * 10))
        self.assertEqual(90, wrap_angle_to_360(90 - 360 * 10))

    def test_wrap_angle_to_180(self):
        for angle in range(-179, 179):
            self.assertEqual(angle, wrap_angle_to_180(angle + 360 * random.randint(-10, 10)))

        for i in range(10):
            self.assertEqual(180, wrap_angle_to_180(180 + 360 * i))

        for i in range(10):
            self.assertEqual(-180, wrap_angle_to_180(-180 - 360 * i))

    def test_heading_to_standard(self):
        # 测试航向角转换为标准单位圆角度的函数
        self.assertEqual(heading_to_standard(0), 90)
        self.assertEqual(heading_to_standard(90), 0)
        self.assertEqual(heading_to_standard(180), 270)
        self.assertEqual(heading_to_standard(-90), 180)
        self.assertEqual(heading_to_standard(-180), 270)

    def test_standard_to_heading(self):
        # 测试标准单位圆角度转换为航向角的函数
        self.assertEqual(90, standard_to_heading(0))  # 0 度转换为航向角为 270 度
        self.assertEqual(0, standard_to_heading(90))  # 90 度转换为航向角为 0 度
        self.assertEqual(-90, standard_to_heading(180))  # 180 度转换为航向角为 90 度
        self.assertEqual(180, standard_to_heading(270))  # 270 度转换为航向角为 180 度


class TestCollisionDetection(unittest.TestCase):
    def test_stationary_intersecting(self):
        """测试两个静止且相交的物体"""
        self.assertTrue(will_collide(np.array([0, 0]), np.array([0, 0]), 1, np.array([0.5, 0]), np.array([0.5, 0]), 1))

    def test_stationary_non_intersecting(self):
        """测试两个静止但不相交的物体"""
        self.assertFalse(will_collide(np.array([0, 0]), np.array([0, 0]), 1, np.array([3, 0]), np.array([3, 0]), 1))

    def test_moving_intersecting(self):
        """测试两个移动且在某个时刻相交的物体"""
        self.assertTrue(will_collide(np.array([0, 0]), np.array([1, 1]), 0.5, np.array([1, 0]), np.array([0, 1]), 0.5))

    def test_moving_non_intersecting(self):
        """测试两个移动但始终不相交的物体"""
        self.assertFalse(will_collide(np.array([0, 0]), np.array([2, 2]), 0.5, np.array([3, 0]), np.array([3, 3]), 0.5))

    def test_zero_relative_velocity(self):
        """测试相对速度为零的情况"""
        self.assertTrue(will_collide(np.array([0, 0]), np.array([0, 0]), 1, np.array([0.5, 0]), np.array([0.5, 0]), 1))
