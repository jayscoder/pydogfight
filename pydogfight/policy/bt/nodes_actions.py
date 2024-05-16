from __future__ import annotations

from pydogfight.policy.bt.common import *


class GoHome(BTPolicyNode):
    """
    行为节点：返回基地
    此节点负责控制代理（如机器人或游戏角色）返回其起始基地或安全区域。此行为通常用于补给、避免危险或结束任务。

    - ALWAYS SUCCESS: 返回基地的操作被设计为总是成功，假定基地是可以到达的，并且代理具有返回基地的能力。
    """

    def updater(self) -> typing.Iterator[Status]:
        home_obj = self.env.battle_area.get_home(self.agent.color)
        yield from go_to_location_updater(self, home_obj.waypoint.location)


class GoToCenter(BTPolicyNode):
    """
    行为节点：飞行到战场中心
    - ALWAYS SUCCESS
    """

    def updater(self) -> typing.Iterator[Status]:
        yield from go_to_location_updater(self, (0, 0))


class GoToSafeArea(BTPolicyNode):
    """
    行为节点：飞行到最近的安全区域，避免过于靠近边界导致毁灭
    """

    def updater(self) -> typing.Iterator[Status]:
        distance = self.agent.distance((0, 0)) - self.env.options.bullseye_safe_radius()
        if distance <= 0:
            # 当前已经在安全区域里了
            yield Status.SUCCESS
            return

        # 朝着中心飞行一定距离
        new_wpt = self.agent.waypoint.move_towards((0, 0), distance + self.agent.speed * 3)
        yield from go_to_location_updater(self, new_wpt.location)


class KeepFly(BTPolicyNode):
    """
    行为节点保持当前航迹继续飞行
    """

    def update(self) -> Status:
        return Status.SUCCESS


class FollowRoute(BTPolicyNode):
    """
    沿着一条预设的航线飞行
    """

    def __init__(self, route: list | str, recursive: bool | str = False, **kwargs):
        super().__init__(**kwargs)
        self.route: list = self.converter.list(route)
        self.route_index = 0
        self.recursive = self.converter.bool(recursive)
        assert len(self.route) > 0

    def updater(self) -> typing.Iterator[Status]:
        while self.recursive:
            for index in range(len(self.route)):
                self.route_index = index
                yield from go_to_location_updater(self, self.route[index])

    def to_data(self):
        return {
            **super().to_data(),
            'route'      : [str(r) for r in self.route],
            'route_index': self.route_index,
            'recursive'  : self.recursive,
            'agent'      : self.agent.to_dict()
        }


class GoToLocation(BTPolicyNode):
    def __init__(self, x: float | str, y: float | str, **kwargs):
        super().__init__(**kwargs)
        self.x = self.converter.float(x)
        self.y = self.converter.float(y)

    def initialise(self) -> None:
        super().initialise()
        self.actions.put_nowait((Actions.go_to_location, self.x, self.y))

    def updater(self) -> typing.Iterator[Status]:
        yield from go_to_location_updater(self, (self.x, self.y))

    def to_data(self):
        return {
            **super().to_data(),
            'x': self.x,
            'y': self.y,
        }


class TurnHeading(BTPolicyNode):

    def __init__(self, heading: float, **kwargs):
        super().__init__(**kwargs)
        self.heading = heading

    def update(self) -> Status:
        h = self.converter.float(self.heading)
        new_wpt = self.agent.waypoint.move(
                d=self.agent.radar_radius,
                angle=h)
        self.actions.put_nowait((Actions.go_to_location, new_wpt.x, new_wpt.y))
        return Status.SUCCESS

# TODO: 条件节点，导弹命中敌机，需要考虑一些匹配
