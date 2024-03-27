from __future__ import annotations

import types
from typing import Tuple, TYPE_CHECKING, Union, Optional

import numpy as np

from pydogfight.core.constants import *
from pydogfight.core.actions import *
import random
from queue import Queue
from pydogfight.utils.rendering import pygame_load_img
from pydogfight.core.models import Waypoint
from pydogfight.algos.traj import calc_optimal_path, OptimalPathParam
from pydogfight.algos.intercept import predict_intercept_point, InterceptPointResult
import math
from pydogfight.utils.rendering import *
import weakref
import json

if TYPE_CHECKING:
    from pydogfight.core.options import Options
    from pydogfight.core.battle_area import BattleArea


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, name: str, options: Options, type: str, color: str = '', x: float = 0, y: float = 0,
                 psi: float = 0,
                 speed: float = 0, turn_radius: float = 0,
                 collision_radius: float = 0):
        assert speed >= 0
        self.name = name
        self.options = options
        self.type = type
        self.color = color
        self.speed = speed
        self.turn_radius = turn_radius
        self.collision_radius = collision_radius

        self.x = x
        self.y = y
        self.psi = psi

        self.destroyed = False  # 是否已经被摧毁

        self.route: np.ndarray | None = None  # 需要遵循的轨迹
        self.route_index: int = -1

        self.actions = Queue()  # 等待消费的行为，每一项是个列表
        self._area = None  # 战场
        # (0, -, -) 0代表什么也不做
        # （1, x, y）飞到指定位置
        # (2, x, y) 朝目标点发射导弹

    def __copy__(self):
        obj = self.__class__(
                name=self.name,
                options=self.options,
                type=self.type,
                color=self.color,
                x=self.x,
                y=self.y,
                psi=self.psi,
                speed=self.speed,
                turn_radius=self.turn_radius)
        obj.destroyed = self.destroyed
        obj.route = self.route
        obj.route_index = self.route_index
        obj._area = self._area
        return obj

    def to_dict(self):
        return {
            'name'            : self.name,
            'type'            : self.type,
            'color'           : self.color,
            'speed'           : self.speed,
            'turn_radius'     : self.turn_radius,
            'collision_radius': self.collision_radius,
            'x'               : self.x,
            'y'               : self.y,
            'psi'             : self.psi,
            'destroyed'       : self.destroyed,
            'route'           : list(self.route),
            'route_index'     : self.route_index
        }

    def __str__(self):
        return json.dump(self.to_dict(), indent=4, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def put_action(self, action):
        if self.destroyed:
            return
        self.actions.put_nowait(action)

    def attach(self, battle_area: 'BattleArea'):
        self._area = weakref.ref(battle_area)

    @property
    def area(self) -> Optional['BattleArea']:
        if self._area is None:
            return None
        return self._area()

    @property
    def waypoint(self) -> Waypoint:
        return Waypoint(self.x, self.y, self.psi)

    @property
    def location(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def screen_position(self) -> tuple[float, float]:
        return game_point_to_screen_point(
                (self.x, self.y),
                game_size=self.options.game_size,
                screen_size=self.options.screen_size)

    def render(self, screen):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def distance(self, to: WorldObj | tuple[float, float]) -> float:
        if isinstance(to, WorldObj):
            to = to.location
        return ((self.x - to[0]) ** 2 + (self.y - to[1]) ** 2) ** 0.5

    def check_collision(self, to: WorldObj):
        # 假设物体是圆形的，可以通过计算它们中心点的距离来检测碰撞
        # 如果两个物体之间的距离小于它们的半径之和，则认为它们发生了碰撞
        if self.collision_radius <= 0 or to.collision_radius <= 0:
            # 其中一方没有碰撞半径的话，就不认为会造成碰撞
            return False
        distance = self.distance(to)
        return distance < (self.collision_radius + to.collision_radius)

    def update(self, delta_time: float):
        pass

    def calc_optimal_path(self, target: tuple[float, float], turn_radius: float) -> OptimalPathParam:
        return calc_optimal_path(
                start=self.waypoint,
                target=target,
                turn_radius=turn_radius
        )

    def on_collision(self, obj: WorldObj):
        """
        与另外一个物体碰撞了
        :param obj: 碰撞的物体
        :return:
        """
        pass

    def check_in_game_range(self):
        if self.options.destroy_on_boundary_exit:
            # 检查是否跑出了游戏范围
            game_x_range = (-self.options.game_size[0] / 2, self.options.game_size[0] / 2)
            game_y_range = (-self.options.game_size[1] / 2, self.options.game_size[1] / 2)

            if self.x < game_x_range[0] or self.x > game_x_range[1]:
                self.destroyed = True
            elif self.y < game_y_range[0] or self.y > game_y_range[1]:
                self.destroyed = True

    def follow_route(self, route) -> bool:
        """
        沿着轨迹运动
        :param route: 轨迹
        :return: 是否运动成功
        """
        if route is None:
            return False
        next_wpt = None
        if isinstance(route, types.GeneratorType):
            try:
                next_wpt = next(route)
            except StopIteration:
                self.route = None
                self.route_index = -1
                return False
        elif len(route) > self.route_index >= 0:
            next_wpt = route[self.route_index]

        if next_wpt is None:
            self.route = None
            self.route_index = -1
            return False
        self.route_index += 1
        self.x = next_wpt[0]
        self.y = next_wpt[1]
        self.psi = next_wpt[2]
        return True

    def move_forward(self, delta_time: float):
        # 朝着psi的方向移动, psi是航向角，0度指向正北，90度指向正东
        # 将航向角从度转换为弧度
        x_theta = self.waypoint.standard_rad
        # 计算 x 和 y 方向上的速度分量
        dx = self.speed * math.cos(x_theta) * delta_time  # 正东方向为正值
        dy = self.speed * math.sin(x_theta) * delta_time  # 正北方向为正值

        # 更新 obj 的位置
        self.x += dx
        self.y += dy

    def move_toward(self, target: tuple[float, float], delta_time: float):
        next_wpt = calc_optimal_path(
                start=self.waypoint,
                target=target,
                turn_radius=self.turn_radius
        ).next_wpt(step=delta_time * self.speed)
        if next_wpt is None:
            return False
        self.x = next_wpt.x
        self.y = next_wpt.y
        self.psi = next_wpt.psi
        return True

    def generate_test_moves(self, angles: list, dis: float) -> list[WorldObj]:
        """
        生成测试的实体（在不同方向上假设实体飞到对应点上）
        :param angles: 测试的不同角度
        :param dis: 飞行的距离
        :return:
        """
        objs: list[WorldObj] = []
        for angle in angles:
            obj_tmp = self.__copy__()
            target_point = (
                self.x + math.cos(math.radians(angle)) * dis,
                self.y + math.sin(math.radians(angle)) * dis,
            )
            obj_tmp.move_toward(target=target_point, delta_time=dis / self.speed)
            objs.append(obj_tmp)
        return objs


class Aircraft(WorldObj):

    def __init__(self,
                 name: str,
                 options: Options,
                 color: str,
                 x: float = 0,
                 y: float = 0,
                 psi: float | None = None,
                 ):
        super().__init__(
                name=name,
                options=options,
                type='aircraft',
                color=color,
                speed=options.aircraft_speed,
                turn_radius=options.aircraft_min_turn_radius,
                x=x,
                y=y,
                psi=psi,
                collision_radius=options.aircraft_missile_count
        )

        self.missile_count = options.aircraft_missile_count  # 剩余的导弹数
        self.fuel = options.aircraft_fuel_capacity  # 飞机剩余油量
        self.fuel_consumption_rate = options.aircraft_fuel_consumption_rate
        self.radar_radius = options.aircraft_radar_radius
        self.missile_destroyed_agents = []  # 导弹摧毁的敌机

    def __copy__(self):
        obj = Aircraft(
                name=self.name,
                options=self.options,
                color=self.color,
                x=self.x,
                y=self.y,
                psi=self.psi)
        obj.destroyed = self.destroyed
        obj.missile_count = self.missile_count
        obj.fuel = self.fuel
        obj.fuel_consumption_rate = self.fuel_consumption_rate
        obj.radar_radius = self.radar_radius
        obj.route = self.route
        obj.route_index = self.route_index
        obj._area = self._area
        obj.missile_destroyed_agents = self.missile_destroyed_agents
        return obj

    def to_dict(self):
        return {
            **super().to_dict(),
            'collision_radius'         : self.collision_radius,
            'missile_count'            : self.missile_count,
            'fuel'                     : self.fuel,
            'fuel_consumption_rate'    : self.fuel_consumption_rate,
            'radar_radius'             : self.radar_radius,
            'missile_destroyed_enemies': self.missile_destroyed_agents
        }

    def render(self, screen):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise e

        assert screen is not None

        render_img(options=self.options,
                   screen=screen,
                   position=self.location,
                   img_path=f'aircraft_{self.color}.svg',
                   label=self.name,
                   rotate=self.psi + 180)
        if self.destroyed:
            render_img(options=self.options,
                       screen=screen,
                       img_path='explosion.svg',
                       position=self.location)

        # 画出导航轨迹
        render_route(options=self.options, screen=screen, route=self.route, color=self.color)

        # 画出雷达圆圈
        render_circle(
                options=self.options,
                screen=screen,
                position=self.location,
                radius=self.radar_radius,
                color='green'
        )

    def update(self, delta_time: float):
        if self.destroyed:
            return
        area = self.area
        if area is None:
            self.destroyed = True
            return

        while not self.actions.empty():
            # 先执行动作
            action = self.actions.get_nowait()
            action_type = int(action[0])
            # print(f'{self.name} 执行动作', Actions(action_type).name)
            if action_type == Actions.go_to_location:
                # 需要移动
                param = calc_optimal_path(self.waypoint, (action[1], action[2]), self.turn_radius)
                self.route = param.generate_traj(delta_time * self.speed)
                self.route_index = 0
            elif action_type == Actions.fire_missile and self.missile_count > 0:
                # 需要发射导弹，以一定概率将对方摧毁
                # 寻找离目标点最近的飞机
                min_dis = float('inf')
                fire_enemy: Aircraft | None = None
                for enemy in area.objs.values():
                    if isinstance(enemy, Aircraft) and enemy.color != self.color:
                        dis = self.distance(enemy)
                        if dis < min_dis and dis < self.radar_radius:
                            # 只能朝雷达范围内的飞机发射导弹
                            min_dis = dis
                            fire_enemy = enemy

                if fire_enemy is not None:
                    self.missile_count -= 1
                    area.add_obj(Missile(source=self, target=fire_enemy, time=area.time))
                    # if random.random() < self.options.predict_missile_hit_prob(self, fire_enemy):
                    #     fire_enemy.destroyed = True
                    #     self.missile_destroyed_agents.append(fire_enemy.name)

        # 执行移动
        if not self.follow_route(route=self.route):
            self.move_forward(delta_time=delta_time)

        # 消耗汽油
        self.fuel -= self.fuel_consumption_rate * delta_time

        # 检查剩余油量
        if self.fuel <= 0:
            self.destroyed = True

        self.check_in_game_range()

    def fire_missile(self, target: tuple[float, float]):
        # 需要发射导弹，以一定概率将对方摧毁
        # 寻找离目标点最近的飞机
        min_dis = float('inf')
        fire_enemy: Aircraft | None = None
        for enemy in self.area.objs.values():
            if isinstance(enemy, Aircraft) and enemy.color != self.color:
                dis = enemy.distance(target)
                if dis < min_dis and dis < self.radar_radius:
                    # 只能朝雷达范围内的飞机发射导弹
                    min_dis = dis
                    fire_enemy = enemy

        if fire_enemy is not None:
            self.missile_count -= 1
            self.area.add_obj(Missile(source=self, target=fire_enemy, time=self.area.time))

    def predict_missile_intercept_point(self, target: Aircraft) -> InterceptPointResult | None:
        """
        预测自己发射的导弹拦截对方的目标点
        :param target:
        :return:
        """
        return predict_intercept_point(
                target=target.waypoint, target_speed=target.speed,
                self_speed=self.options.missile_speed,
                calc_optimal_dis=lambda p: calc_optimal_path(
                        start=self.waypoint,
                        target=(target.x, target.y),
                        turn_radius=self.options.missile_min_turn_radius
                ).length)

    def predict_aircraft_intercept_point(self, target: Aircraft) -> InterceptPointResult | None:
        """
        预测自己拦截敌方的目标点
        :param target: 自己是飞机
        :return:
        """
        return predict_intercept_point(
                target=target.waypoint, target_speed=target.speed,
                self_speed=self.speed,
                calc_optimal_dis=lambda p: calc_optimal_path(
                        start=self.waypoint,
                        target=(target.x, target.y),
                        turn_radius=self.turn_radius
                ).length)

    def on_collision(self, obj: WorldObj):
        if self.destroyed:
            return
        if isinstance(obj, Aircraft):
            obj.destroyed = True
            self.destroyed = True

    def in_radar_range(self, obj: WorldObj) -> bool:
        """
        obj是否在雷达范围内
        :param obj:
        :return:
        """
        return self.distance(obj) <= self.radar_radius


class Missile(WorldObj):
    def __init__(self, source: Aircraft, target: Aircraft, time: float):
        """
        :param source:
        :param target:
        """
        super().__init__(
                type='missile',
                options=source.options,
                name=f'{source.name}_missile_{source.missile_count}',
                color=source.color,
                speed=source.options.missile_speed,
                turn_radius=source.options.missile_min_turn_radius,
                x=source.x,
                y=source.y,
                psi=source.psi
        )
        self.collision_radius = self.options.missile_collision_radius
        self.turn_radius = self.options.missile_min_turn_radius
        self.fire_time = time  # 发射的时间
        self.source = source
        self.target = target
        self.fuel = self.options.missile_fuel_capacity
        self.fuel_consumption_rate = self.options.missile_fuel_consumption_rate

        self._last_generate_route_time = 0

    def render(self, screen):
        # print('render missile', self.name, self.screen_position, self.destroyed)
        if self.destroyed:
            return

        render_img(
                options=self.options,
                screen=screen,
                position=self.location,
                img_path=f'missile_{self.color}.svg',
                # label=self.name,
                rotate=self.psi + 90,
        )

        render_route(
                options=self.options,
                screen=screen,
                route=self.route,
                color=self.color,
                count=20
        )

    def update(self, delta_time: float):
        if self.destroyed:
            self.area.remove_obj(self)
            return

        self.fuel -= self.fuel_consumption_rate * delta_time

        if self.fuel <= 0:
            self.destroyed = True
            return

        if self.area.time - self._last_generate_route_time > self.options.missile_reroute_interval:
            self._last_generate_route_time = self.area.time
            # 每隔1秒重新生成一次轨迹
            hit_param = calc_optimal_path(
                    start=self.waypoint,
                    target=self.target.location,
                    turn_radius=self.turn_radius
            )
            if hit_param.length != float('inf'):
                self.route = hit_param.generate_traj(step=delta_time * self.speed)  # 生成轨迹
                self.route_index = 0

        if not self.follow_route(route=self.route):
            self.move_forward(delta_time=delta_time)

        self.check_in_game_range()

    def on_collision(self, obj: WorldObj):
        if self.destroyed:
            return

        if isinstance(obj, Aircraft):
            if obj == self.source:
                return
            self.destroyed = True
            obj.destroyed = True
            self.source.missile_destroyed_agents.append(obj.name)

    def predict_aircraft_intercept_point(self, target: Aircraft) -> InterceptPointResult | None:
        """
        预测自己拦截敌方的目标点（自己是导弹）
        :param target: 需要攻击的目标飞机
        :return:
        """
        return predict_intercept_point(
                target=target.waypoint, target_speed=target.speed,
                self_speed=self.speed,
                calc_optimal_dis=lambda p: calc_optimal_path(
                        start=self.waypoint,
                        target=target.location,
                        turn_radius=self.turn_radius
                ).length)


class Home(WorldObj):
    def __init__(self, name: str, color: str, options: Options, x: float, y: float):
        super().__init__(type='home', options=options, name=name, color=color, x=x, y=y)
        self.radius = options.home_area_radius

    def render(self, screen):
        render_img(options=self.options,
                   screen=screen,
                   position=self.location,
                   img_path=f'home_{self.color}.svg',
                   label=self.name)
        # 画出安全圆圈
        render_circle(options=self.options, screen=screen, position=self.location, radius=self.radius, color='green')

    def update(self, delta_time: float):
        # 查看哪些飞机飞到了基地附近
        area = self.area

        if area is None:
            self.destroyed = True
            return

        for obj in area.objs.values():
            if not isinstance(obj, Aircraft):
                continue
            dis = self.distance(obj)
            if dis < self.radius:
                if obj.color == self.color:
                    # 加油
                    if self.options.home_refuel and obj.fuel < self.options.home_refuel_threshold_capacity:
                        obj.fuel = self.options.aircraft_fuel_capacity
                    if self.options.home_replenish_missile and obj.missile_count < self.options.home_replenish_missile_threshold_count:
                        obj.missile_count = self.options.aircraft_missile_count
                else:
                    # 摧毁敌机
                    if self.options.home_attack:
                        obj.destroyed = True

# class MousePoint(WorldObj):
#     def __init__(self, point: tuple[float, float], time: float):
#         super().__init__(type='mouse_point', options=Options(), name=f'mouse_point_{point[0]}_{point[1]}')
#         self.point = point
#         self.time = time
#
#     def update(self, area: BattleArea, delta_time: float):
#         if area.duration - self.time > self.options.missile_render_duration:
#             self.destroyed = True
#         if self.destroyed:
#             area.remove_obj(self)
#
#     def render(self, screen):
#         if self.destroyed:
#             return
#         import pygame
#         pygame.draw.circle(screen, COLORS[self.color], self.point, 5)  # 绘制鼠标点
