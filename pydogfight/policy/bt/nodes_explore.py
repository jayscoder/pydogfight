from __future__ import annotations

from pydogfight.policy.bt.common import *
import random


class Explore(BTPolicyNode):
    """
    行为节点：探索未知区域
    此节点负责控制代理（如机器人或游戏角色）探索它尚未访问的地方。节点的目标是扩大代理的知识范围，通过探索环境来发现新的区域或点。

    - SUCCESS: 探索成功，表示代理成功移动到一个之前未探索的位置。
    - FAILURE: 探索失败，可能因为以下原因：
        - 代理当前有尚未完成的路线或任务，因此无法开始新的探索。
        - 环境中没有更多的未探索区域，或者无法从当前位置移动到未探索的区域。
    """

    def updater(self) -> typing.Iterator[Status]:
        go_to_location = self.agent.position_memory.pick_position()
        yield from go_to_location_updater(self, go_to_location)


class GridExplore(BTPolicyNode):
    """
    网格式探索，将整个战场分成了W*H的网格，然后探索其中的第index个
    index: 探索的索引
    - random: 随机
    - int: 具体的某个索引
    - {{}} # 从context里获取
    """

    def __init__(self, W: int | str, H: int | str, index: int | str = 'random', **kwargs):
        super().__init__(**kwargs)
        self.W = W
        self.H = H
        self.index = index
        self.centers: list[tuple[float, float]] = []  # 存储网格中心的坐标
        self.grid_width = 0  # 网格宽度
        self.grid_height = 0  # 网格高度
        self.curr_index = None

    @classmethod
    def compute_grid_centers(cls, W: int, H: int, game_size: tuple[float, float]):
        centers = []
        grid_width = game_size[0] / W
        grid_height = game_size[1] / H
        for hi in range(H):
            for wi in range(W):
                centers.append(
                        ((wi + 0.5) * grid_width - game_size[0] / 2, (hi + 0.5) * grid_height - game_size[1] / 2),
                )
        return centers

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.W = self.converter.int(self.W)
        self.H = self.converter.int(self.H)
        self.grid_width = self.env.options.game_size[0] / self.W
        self.grid_height = self.env.options.game_size[1] / self.H
        self.centers = self.compute_grid_centers(W=self.W, H=self.H, game_size=self.env.options.game_size)

    def to_data(self):
        return {
            **super().to_data(),
            "centers"    : self.centers,
            "grid_width" : self.grid_width,
            "grid_height": self.grid_height,
            'index'      : self.curr_index,
            'W'          : self.W,
            'H'          : self.H
        }

    def cal_index(self) -> int:
        if isinstance(self.index, int):
            return self.index
        elif self.index == 'random':
            return random.randint(0, len(self.centers) - 1)
        else:
            return self.converter.int(self.index)

    def updater(self) -> typing.Iterator[Status]:
        index = self.cal_index()
        self.curr_index = index
        go_to_location = self.centers[index]
        yield from go_to_location_updater(self, go_to_location)



