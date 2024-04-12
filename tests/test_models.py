import unittest
from pydogfight.utils.models import *


class TestWaypoint(unittest.TestCase):

    def test_move(self):
        self.assertEqual(Waypoint.build(x=0, y=10, psi=0), Waypoint.build(x=0, y=0, psi=0).move(10, angle=0))

        self.assertEqual(Waypoint.build(x=10, y=0, psi=90), Waypoint.build(x=0, y=0, psi=90).move(10, angle=0))

        self.assertEqual(Waypoint.build(x=10, y=0, psi=90), Waypoint.build(x=0, y=0, psi=0).move(10, angle=90))

    def test_move_towards(self):
        self.assertEqual(Waypoint.build(0, 9, 0), Waypoint.build(0, 0, 0).move_towards((0, 10), 9))
        self.assertEqual(Waypoint.build(10, 0, 90), Waypoint.build(0, 0, 0).move_towards((10, 0), 100))
        self.assertEqual(Waypoint.build(0, 0, 0), Waypoint.build(0, 0, 0).move_towards((0, 0), 9))

