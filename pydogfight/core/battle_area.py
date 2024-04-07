from __future__ import annotations

from pydogfight.core.world_obj import *
from pydogfight.core.options import Options
from collections import defaultdict


class BattleArea:
    def __init__(self, options: Options, render_mode: str = 'rgb_array'):
        self.options = options
        self.size = options.game_size
        self.time = 0  # 对战累积时长
        self.objs: dict[str, WorldObj] = { }
        self.render_mode = render_mode

    def reset(self):
        self.time = 0
        self.objs = { }

        home_position = self.options.generate_home_init_position()
        assert self.options.red_home != ''
        assert self.options.blue_home != ''
        self.add_obj(
                Home(
                        name=self.options.red_home,
                        options=self.options,
                        color='red',
                        x=home_position[0][0],
                        y=home_position[0][1],
                )
        )

        self.add_obj(
                Home(
                        name=self.options.blue_home,
                        options=self.options,
                        color='blue',
                        x=home_position[1][0],
                        y=home_position[1][1],
                )
        )

        self.add_obj(
            Bullseye(options=self.options)
        )

        for name in self.options.red_agents:
            # 随机生成飞机位置
            wpt = self.options.generate_aircraft_init_waypoint(home_position[0], self.options.home_area_radius)
            self.add_obj(Aircraft(
                    name=name,
                    options=self.options,
                    color='red',
                    x=wpt[0],
                    y=wpt[1],
                    psi=wpt[2]))

        for name in self.options.blue_agents:
            # 随机生成飞机位置
            wpt = self.options.generate_aircraft_init_waypoint(home_position[1], self.options.home_area_radius)
            self.add_obj(Aircraft(
                    name=name,
                    options=self.options,
                    color='blue',
                    x=wpt[0],
                    y=wpt[1],
                    psi=wpt[2]))

    def add_obj(self, obj: WorldObj):
        self.objs[obj.name] = obj
        obj.attach(battle_area=self)

    def get_obj(self, name: str) -> WorldObj:
        return self.objs[name]

    def remove_obj(self, obj: WorldObj) -> None:
        del self.objs[obj.name]

    @property
    def agents(self) -> list[Aircraft]:
        return list(filter(lambda agent: isinstance(agent, Aircraft), self.objs.values()))

    @property
    def missiles(self) -> list[Missile]:
        return list(filter(lambda obj: isinstance(obj, Missile), self.objs.values()))

    @property
    def homes(self) -> list[Home]:
        return list(filter(lambda obj: isinstance(obj, Home), self.objs.values()))

    @property
    def bullseye(self) ->  Bullseye:
        obj = self.objs.get('bullseye')
        assert isinstance(obj, Bullseye)
        return obj

    def render(self, screen):
        for obj in self.objs.values():
            obj.render(screen)

    ### 战场环境更新，每一轮每个物体只消费一个行为 ###
    def update(self):
        """
        :return:
        """
        for obj in list(self.objs.values()):
            if obj.destroyed:
                continue
            obj.update(delta_time=self.options.delta_time)

        # 检查碰撞
        obj_list = list(self.objs.values())
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                if obj_list[i].destroyed or obj_list[j].destroyed:
                    continue
                if obj_list[i].check_collision(obj_list[j]):
                    obj_list[i].on_collision(obj_list[j])
                    obj_list[j].on_collision(obj_list[i])

        self.time += self.options.delta_time

    @property
    def remain_count(self) -> dict:
        """
        获取剩余实体数量
        :return:
        """
        count = {
            'aircraft': defaultdict(int),
            'missile' : defaultdict(int),
            'home'    : defaultdict(int)
        }
        for obj in self.objs.values():
            if obj.destroyed:
                continue
            if obj.type not in count:
                count[obj.type] = defaultdict(int)
            count[obj.type][obj.color] += 1
        return count

    def find_nearest_enemy(self, agent_name: str, ignore_radar: bool = False) -> Aircraft | None:
        """
        找到距离最近的敌人
        :param agent_name:
        :param ignore_radar: 是否忽略雷达因素（设为false则只会返回雷达范围内的敌机）
        :return:
        """
        agent = self.objs[agent_name]
        assert isinstance(agent, Aircraft)
        min_dis = float('inf')
        enemy = None
        for obj in self.objs.values():
            if obj.destroyed or not isinstance(obj, Aircraft):
                continue
            if obj.color == agent.color:
                continue

            dis = obj.distance(agent)
            if not ignore_radar and dis > agent.radar_radius:
                continue

            if dis < min_dis:
                min_dis = dis
                enemy = obj

        return enemy
