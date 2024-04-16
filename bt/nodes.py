from __future__ import annotations
from bt.base import *

class RandomInitWaypointNearGameCenter(BTPolicyNode):
    """
    随机初始化自己的位置在战场中心雷达半径附近，方便双方一开始就能相遇在双方的雷达半径内
    Randomly initializes the agent's position near the center of the battlefield radar radius,
    facilitating an early encounter within each other's radar range.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inited = False

    def reset(self):
        super().reset()
        self.inited = False

    def update(self):
        if self.inited:
            return Status.SUCCESS
        self.inited = True
        if self.agent.color == 'red':
            x = -self.agent.radar_radius / 2
        else:
            x = self.agent.radar_radius / 2
        psi = random.random() * 360

        self.agent.waypoint = Waypoint.build(x=x, y=0, psi=psi)

        return Status.SUCCESS


class CheatGoToNearestEnemy(BTPolicyNode):
    """
    开挂：朝着敌机飞
    """

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.detect_aircraft(agent_name=self.agent_name, ignore_radar=True, only_enemy=True)
        if len(enemy) == 0:
            yield Status.FAILURE
            return

        yield from go_to_location_updater(self, location=enemy[0].waypoint.location)


class CheatGoToNearestEnemyWithMemory(BTPolicyNode):
    """
    开挂：朝着敌机的记忆位置飞
    只要改记忆位置在敌机的雷达范围内，就不更改这个位置
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_location = None

    def reset(self):
        super().reset()
        self.memory_location = None

    def updater(self) -> typing.Iterator[Status]:
        enemy = self.env.battle_area.find_nearest_enemy(agent_name=self.agent_name, ignore_radar=True)
        if enemy is None:
            yield Status.FAILURE
            return

        if self.memory_location is not None and enemy.in_radar_range(self.memory_location):
            # self.actions.put_nowait((Actions.go_to_location, location[0], location[1]))
            yield from go_to_location_updater(self, location=self.memory_location)
            return

        self.memory_location = enemy.waypoint.location
        yield from go_to_location_updater(self, location=self.memory_location)


class TestNode(BTPolicyNode):
    """定义的新的节点，需要在builder中注册才能在xml中使用"""

    def __init__(self, count: int | str = '', **kwargs):
        super(TestNode, self).__init__(**kwargs)  # 必须得给父节点传递kwargs参数，不然会报错
        self.count = count  # 自定义的参数在使用xml构建的时候传进来的一定是str，后面需要在setup的时候通过converter进行数值转换

    def setup(self, **kwargs: typing.Any) -> None:
        super(TestNode, self).setup(**kwargs)
        self.count = self.converter.int(self.count)
        # jinjia2语法需要通过self.converter.render(...)来进行转换

    def to_data(self):
        # 这里的数据会在track的时候保存下来
        return {
            **super().to_data(),
            'agent': self.agent.to_dict()
        }




