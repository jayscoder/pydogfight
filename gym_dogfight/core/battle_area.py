from __future__ import annotations

from gym_dogfight.core.world_obj import *
from gym_dogfight.algos.traj import calc_optimal_path, Waypoint
from gym_dogfight.core.options import Options
from gym_dogfight.core.actions import Actions
import math


class BattleArea:
    def __init__(self, options: Options, render_mode: str = 'rgb_array'):
        self.options = options
        self.size = options.game_size
        self.duration = 0  # 对战累积时长
        self.objs: dict[str, WorldObj] = { }
        self.render_mode = render_mode

    def reset(self):
        self.duration = 0
        self.objs = { }

        home_position = self.options.generate_home_init_position()
        if self.options.red_home != '':
            self.add_obj(
                    Home(
                            name=self.options.red_home,
                            options=self.options,
                            color='red',
                            x=home_position[0][0],
                            y=home_position[0][1],
                    )
            )

        if self.options.blue_home != '':
            self.add_obj(
                    Home(
                            name=self.options.blue_home,
                            options=self.options,
                            color='blue',
                            x=home_position[1][0],
                            y=home_position[1][1],
                    )
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

    def render(self, screen):
        for obj in self.objs.values():
            obj.render(screen)

    ### 战场环境更新，每一轮每个物体只消费一个行为 ###
    def update(self, delta_time: float = 0.1):
        """

        :param delta_time: 间隔时间
        :return:
        """
        if delta_time <= 0:
            return

        for obj in list(self.objs.values()):
            if obj.destroyed:
                continue
            obj.update(delta_time=delta_time)

        # 检查碰撞
        obj_list = list(self.objs.values())
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                if obj_list[i].destroyed or obj_list[j].destroyed:
                    continue
                if obj_list[i].check_collision(obj_list[j]):
                    obj_list[i].destroyed = True
                    obj_list[j].destroyed = True


        self.duration += delta_time

    @property
    def remain_count(self) -> (int, int):
        """
        获取双方剩余战机数量
        :return:
        """
        red_count = 0
        blue_count = 0

        for agent in self.agents:
            if agent.destroyed:
                continue
            if agent.color == 'red':
                red_count += 1
            elif agent.color == 'blue':
                blue_count += 1
        return red_count, blue_count

    def find_nearest_enemy(self, agent_name: str, ignore_radar: bool = False) -> Aircraft | None:
        """
        找到距离最近的敌人
        :param agent_name:
        :param ignore_radar:
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
