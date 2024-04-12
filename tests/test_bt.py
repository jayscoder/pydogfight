from pydogfight.policy.bt import *
import unittest


class TestGridExplore(unittest.TestCase):

    def test_grid_explore(self):
        centers = GridExplore.compute_grid_centers(
                W=2,
                H=2,
                game_size=(1000, 1000)
        )
        # 检查计算的中心坐标数量是否符合预期
        self.assertEqual(len(centers), 4)

        self.assertEqual(centers, [
            (-250, -250),
            (250, -250),
            (-250, 250),
            (250, 250)
        ])

