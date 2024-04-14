import unittest
from pydogfight.utils.models import *


class TestWaypoint(unittest.TestCase):
    def setUp(self):
        # 初始化一个 Waypoint 对象
        self.wpt = Waypoint.build(x=0, y=0, psi=0)  # 假设航向0度指向正东

    def test_move_forward_no_turn(self):
        """测试无转弯直线移动"""
        new_wpt = self.wpt.move(10, 0)  # 向正北移动10单位距离
        self.assertAlmostEqual(new_wpt.x, 0)
        self.assertAlmostEqual(new_wpt.y, 10)
        self.assertAlmostEqual(new_wpt.psi, 0)

    def test_move_with_positive_angle(self):
        """测试正角度转向移动"""
        new_wpt = self.wpt.move(10, 90)  # 向正东移动10单位距离
        self.assertAlmostEqual(new_wpt.x, 10)
        self.assertAlmostEqual(new_wpt.y, 0)
        self.assertAlmostEqual(new_wpt.psi, 90)


    def test_move_with_negative_angle(self):
        """测试负角度转向移动"""
        new_wpt = self.wpt.move(10, -90)
        self.assertAlmostEqual(new_wpt.x, -10)
        self.assertAlmostEqual(new_wpt.y, 0)
        self.assertAlmostEqual(new_wpt.psi, -90)  # 新航向应为-90度

    def test_move_complex(self):
        """测试复杂角度和距离"""
        # 初始化在正东方向，然后向正南方向移动5单位，并转180度
        self.wpt.psi = 180  # 初始航向正南
        new_wpt = self.wpt.move(5, 180)
        self.assertAlmostEqual(new_wpt.x, 0)
        self.assertAlmostEqual(new_wpt.y, 5)  # 向南移动5单位
        self.assertAlmostEqual(new_wpt.psi, 0)

    def test_move(self):
        self.assertEqual(Waypoint.build(x=0, y=10, psi=0), Waypoint.build(x=0, y=0, psi=0).move(10, angle=0))

        self.assertEqual(Waypoint.build(x=10, y=0, psi=90), Waypoint.build(x=0, y=0, psi=90).move(10, angle=0))

        self.assertEqual(Waypoint.build(x=10, y=0, psi=90), Waypoint.build(x=0, y=0, psi=0).move(10, angle=90))

    def test_move_towards(self):
        self.assertEqual(Waypoint.build(0, 9, 0), Waypoint.build(0, 0, 0).move_towards((0, 10), 9))
        self.assertEqual(Waypoint.build(10, 0, 90),
                         Waypoint.build(0, 0, 0).move_towards((10, 0), 100, allow_over=False))
        self.assertEqual(Waypoint.build(0, 0, 0), Waypoint.build(0, 0, 0).move_towards((0, 0), 9))
