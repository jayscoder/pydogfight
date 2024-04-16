from __future__ import annotations

from typing import Tuple, TYPE_CHECKING, Union, Optional

import numpy as np

from pydogfight.core.actions import *
from queue import Queue
from pydogfight.utils.models import Waypoint
from pydogfight.utils.traj import calc_optimal_path, OptimalPathParam
from pydogfight.utils.intercept import *
from pydogfight.utils.rendering import *
import weakref
from pydogfight.utils.position_memory import PositionMemory
from pydogfight.utils.models import *
from pydogfight.utils.common import will_collide

if TYPE_CHECKING:
    from pydogfight.core.options import Options
    from pydogfight.core.battle_area import BattleArea


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self,
                 name: str,
                 options: Options,
                 type: str,
                 waypoint: Waypoint,
                 color: str = '',
                 speed: float = 0,
                 turn_radius: float = 0,
                 collision_radius: float = 0):
        assert speed >= 0
        self.name = name
        self.options = options
        self.type = type
        self.color = color

        self.turn_radius = turn_radius
        self.collision_radius = collision_radius

        self.speed = speed
        self.waypoint = waypoint

        self.indestructible = False  # 是否不可摧毁
        self.destroyed = False  # 是否已经被摧毁
        self.destroyed_reason = []  # 被摧毁的原因
        self.destroyed_count = 0  # 被摧毁次数
        self.destroyed_time = 0  # 被摧毁的时间

        self.waiting_actions = Queue(maxsize=10)  # 等待消费的行为，每一项是个列表
        self.consumed_actions = Queue(maxsize=10)  # 最近消费的10个动作

        self._area = None  # 战场
        self.last_is_in_game_range = True  # 用来保存之前是否在游戏区域，避免超出游戏区域后每次都触发摧毁自己
        self.last_waypoint: Waypoint | None = None  # 上一刻的航迹点

        self.route_param: None | OptimalPathParam = None
        self.route_param_time = 0
        self.render_route: np.ndarray | None = None  # 需要遵循的轨迹

        # (0, -, -) 0代表什么也不做
        # （1, x, y）飞到指定位置
        # (2, x, y) 朝目标点发射导弹

    def __copy__(self):
        obj = self.__class__(
                name=self.name,
                options=self.options,
                type=self.type,
                color=self.color,
                waypoint=self.waypoint.__copy__(),
                speed=self.speed,
                turn_radius=self.turn_radius)
        obj.copy_from(self)
        return obj

    def copy_from(self, obj: WorldObj):
        self.name = obj.name
        self.options = obj.options
        self.type = obj.type
        self.color = obj.color
        self.turn_radius = obj.turn_radius
        self.collision_radius = obj.collision_radius

        self.speed = obj.speed
        self.waypoint = obj.waypoint.__copy__()
        self.indestructible = obj.indestructible
        self.destroyed = obj.destroyed
        self.destroyed_count = obj.destroyed_count
        self.destroyed_reason = obj.destroyed_reason.copy()
        self.destroyed_time = obj.destroyed_time

        self.render_route = obj.render_route
        self.route_param = obj.route_param
        self.route_param_time = obj.route_param_time
        self._area = obj._area

    def to_dict(self):
        return {
            'name'            : self.name,
            'type'            : self.type,
            'color'           : self.color,
            'speed'           : self.speed,
            'turn_radius'     : self.turn_radius,
            'collision_radius': self.collision_radius,
            'indestructible'  : self.indestructible,
            'waypoint'        : self.waypoint,
            'destroyed'       : self.destroyed,
            'destroyed_reason': self.destroyed_reason,
            'destroyed_count' : self.destroyed_count,
            'destroyed_time'  : self.destroyed_time,
            'survival_time'   : self.survival_time,
            'waiting_actions' : [str(act) for act in read_queue_without_destroying(self.waiting_actions)],
            'consumed_actions': [str(act) for act in read_queue_without_destroying(self.consumed_actions)],
            'is_in_game_range': self.is_in_game_range
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def destroy(self, reason: str, source=None):
        if not self.indestructible:
            self.destroyed = True
        if self.waiting_actions == 0:
            self.destroyed_time = self.area.time
        self.destroyed_count += 1
        self.destroyed_reason.append((reason, source))

    def put_action(self, action):
        if self.destroyed:
            return
        if int(action[0]) == Actions.keep:
            return
        if self.waiting_actions.full():
            self.waiting_actions.get_nowait()
        self.waiting_actions.put_nowait(action)

    def attach(self, battle_area: 'BattleArea'):
        self._area = weakref.ref(battle_area)

    @property
    def area(self) -> Optional['BattleArea']:
        if self._area is None:
            return None
        return self._area()

    @property
    def enemy_color(self):
        """敌方战队的颜色"""
        if self.color == 'red':
            return 'blue'
        else:
            return 'red'

    @property
    def survival_time(self) -> float:
        """存活时间"""
        if self.destroyed_time > 0:
            return self.destroyed_time
        else:
            return self.area.time

    @property
    def screen_position(self) -> tuple[float, float]:
        return game_point_to_screen_point(
                game_point=self.waypoint.location,
                game_size=self.options.game_size,
                screen_size=self.options.screen_size)

    def render(self, screen):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def distance(self, to: WorldObj | tuple[float, float] | Waypoint | np.ndarray) -> float:
        if isinstance(to, WorldObj):
            to = to.waypoint
        return self.waypoint.distance(to)

    def will_collide(self, obj: WorldObj):
        # 假设物体是圆形的，可以通过计算它们中心点的距离来检测碰撞
        # 如果两个物体之间的距离小于它们的半径之和，则认为它们发生了碰撞
        if self.collision_radius <= 0 or obj.collision_radius <= 0:
            # 其中一方没有碰撞半径的话，就不认为会造成碰撞
            return False
        collide_ratio = self.distance(obj) / (self.collision_radius + obj.collision_radius)
        if collide_ratio < 1:
            return True

        obj_last_wpt = obj.last_waypoint or obj.waypoint
        self_last_wpt = self.last_waypoint or self.waypoint

        collided = will_collide(
                a1=self_last_wpt.location,
                a2=self.waypoint.location,
                b1=obj_last_wpt.location,
                b2=obj.waypoint.location,
                ra=self.collision_radius,
                rb=obj.collision_radius
        )
        # if collided and distance > 1:
        #     print('碰撞异常', distance)
        return collided

    def update(self, delta_time: float):
        """更新状态"""
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

    @property
    def is_in_game_range(self) -> bool:
        # 检查是否跑出了游戏范围
        game_x_range = (-self.options.game_size[0] / 2, self.options.game_size[0] / 2)
        game_y_range = (-self.options.game_size[1] / 2, self.options.game_size[1] / 2)
        boundary = BoundingBox.from_range(x_range=game_x_range, y_range=game_y_range)
        return boundary.contains(self.waypoint.location)

    def do_move(self, waypoint: Waypoint):
        self.last_waypoint = self.waypoint
        self.waypoint = waypoint

    def update_follow_route(self) -> bool:
        """
        沿着轨迹运动
        :return: 是否运动成功
        """
        if self.route_param is None:
            return False
        move_length = self.speed * (self.area.time - self.route_param_time)
        if move_length > self.route_param.length:
            self.render_route = None
            self.route_param = None
            self.route_param_time = 0
            return False

        next_wpt = self.route_param.next_waypoint(length=self.speed * (self.area.time - self.route_param_time))
        if next_wpt is None:
            self.render_route = None
            self.route_param = None
            self.route_param_time = 0
            return False
        self.do_move(next_wpt)
        return True

    def update_move_forward(self, delta_time: float):
        # 朝着psi的方向移动, psi是航向角，0度指向正北，90度指向正东
        # 将航向角从度转换为弧度
        new_wpt = self.waypoint.move(d=delta_time * self.speed, angle=0)
        self.do_move(new_wpt)

    def go_to_location(self, target: tuple[float, float]):
        # if self.route is not None and len(self.route) > 0 and not force:
        #     # 当前还有未完成的路由
        #     route_target = self.route[-1][:2]
        #     target_distance = np.linalg.norm(route_target - np.array(target))
        #     # print(f'{self.name} target_distance', target_distance)
        #     if target_distance <= 1:
        #         # print('目标点和自己当前的路由终点很接近，就不再重复导航')
        #         # 如果目标点和自己当前的路由终点很接近，就不再重复导航了
        #         return

        # TODO: 当前的状态可能是在拐弯，计算最佳路径的时候需要考虑进去
        # 需要移动
        # 判断一下当前路由的终点是哪里
        route_param = calc_optimal_path(start=self.waypoint, target=target, turn_radius=self.turn_radius)
        if route_param.length == 0 or route_param.length == float('inf'):
            return
        self.route_param = route_param
        self.route_param_time = self.area.time - self.options.delta_time  # 需要通过这种方式来保证当前帧就能更新位置，不然会丢掉一帧
        if self.options.render:
            self.render_route = self.route_param.build_route(self.route_param.length / 20)

    def generate_test_moves(self, in_safe_area: bool = True, angle_sep: int = 45) -> list[Waypoint]:
        """
        生成测试的实体的预期位置（在不同方向上假设实体飞到对应点上）
        :param in_safe_area: 是否强制要求在安全位置（不能离战场中心远）
        :param angle_sep: 角度的间隔
        :return:
        """
        # 测试的距离
        dis = max(self.turn_radius * 10, self.speed * 20)
        wpt_list: list[Waypoint] = []
        for angle in range(0, 360, angle_sep):
            target = self.waypoint.move(d=dis, angle=angle)
            if in_safe_area and target.distance([0, 0]) >= self.options.bullseye_safe_radius():
                # 距离战场中心太远了
                continue
            new_wpt = self.waypoint.optimal_move_towards(
                    target=target.location, d=dis,
                    turn_radius=self.turn_radius)
            wpt_list.append(new_wpt)
        return wpt_list

    def is_reach_location(self, p: tuple[float, float]) -> bool:
        return self.distance(p) <= self.speed * self.options.reach_location_interval()

    def is_current_route_target(self, target: tuple[float, float]) -> bool:
        # 是否是当前路由的目标
        if self.route_param is None:
            return self.is_reach_location(target)
        return self.is_reach_location(self.route_param.target.location)

    def predict_intercept_point(
            self,
            target_wpt: Waypoint,
            target_speed: float) -> InterceptPointResult | None:
        """
        预测自己拦截敌方的目标点
        :param target_wpt: 需要攻击的目标飞机的航迹点
        :param target_speed: 需要攻击的目标飞机的速度
        :return:
        """
        return predict_intercept_point(
                target=target_wpt, target_speed=target_speed,
                self_speed=self.speed,
                calc_optimal_dis=lambda p: calc_optimal_path(
                        start=self.waypoint,
                        target=target_wpt,
                        turn_radius=self.turn_radius
                ).length)


class Aircraft(WorldObj):

    def __init__(self,
                 name: str,
                 options: Options,
                 color: str,
                 waypoint: Waypoint
                 ):
        super().__init__(
                name=name,
                options=options,
                type='aircraft',
                color=color,
                speed=options.aircraft_speed,
                turn_radius=options.aircraft_min_turn_radius,
                waypoint=waypoint,
                collision_radius=options.aircraft_collision_radius
        )

        self.missile_count = options.aircraft_missile_count  # 剩余的导弹数
        self.fuel = options.aircraft_fuel_capacity  # 飞机剩余油量
        self.fuel_consumption_rate = options.aircraft_fuel_consumption_rate
        self.radar_radius = options.aircraft_radar_radius
        self.missile_hit_enemy = []  # 自己发射的导弹命中的战机
        self.missile_hit_enemy_count = 0
        self.missile_hit_self = []  # 自己被导弹命中的次数
        self.missile_hit_self_count = 0
        self.missile_evade_success = []  # 自己规避的导弹
        self.missile_evade_success_count = 0  # 自己规避导弹成功次数
        self.missile_miss = []  # 自己发射的哪些导弹没有命中敌机
        self.missile_miss_count = 0  # 自己发射的导弹没有命中敌机的次数

        self.missile_fire_fail_count = 0  # 发射导弹失败的次数

        self.aircraft_collided_count = 0  # 与飞机相撞次数

        self.last_fire_missile_time = 0  # 上次发射导弹时间

        self.fuel_depletion_count = 0  # 燃油耗尽次数
        self.missile_depletion_count = 0  # 导弹耗尽次数
        self.indestructible = options.indestructible

        self.home_returned_count = 0  # 到基地次数

        self.position_memory = PositionMemory(
                boundary=self.options.safe_boundary(),
                sep=self.options.aircraft_position_memory_sep)

        self.missile_fired = []  # 已经发射过的导弹
        self.missile_fired_count = 0  # 发射过的导弹数量

        # self.detected_enemy_count = 0  # 检查到的敌人的数量

    def __copy__(self):
        obj = Aircraft(
                name=self.name,
                options=self.options,
                color=self.color,
                waypoint=self.waypoint.__copy__())
        obj.copy_from(self)
        return obj

    def copy_from(self, obj: Aircraft):
        super().copy_from(obj)
        self.missile_count = obj.missile_count  # 剩余的导弹数
        self.fuel = obj.fuel  # 飞机剩余油量
        self.fuel_consumption_rate = obj.fuel_consumption_rate
        self.radar_radius = obj.radar_radius
        self.missile_hit_enemy = obj.missile_hit_enemy.copy()  # 自己发射的导弹命中的战机
        self.missile_hit_enemy_count = obj.missile_hit_enemy_count
        self.missile_hit_self = obj.missile_hit_self.copy()  # 自己被导弹命中的次数
        self.missile_hit_self_count = obj.missile_hit_self_count
        self.missile_evade_success = obj.missile_evade_success  # 自己规避的导弹
        self.missile_evade_success_count = obj.missile_evade_success_count  # 自己规避导弹成功次数

        self.missile_miss = obj.missile_miss.copy()  # 自己发射的哪些导弹没有命中敌机
        self.missile_miss_count = obj.missile_miss_count  # 自己发射的导弹没有命中敌机的次数
        self.aircraft_collided_count = obj.aircraft_collided_count

        self.last_fire_missile_time = obj.last_fire_missile_time  # 上次发射导弹时间

        self.fuel_depletion_count = obj.fuel_depletion_count  # 燃油耗尽次数
        self.missile_depletion_count = obj.fuel_depletion_count

        self.position_memory = obj.position_memory

        self.missile_fired = obj.missile_fired.copy()
        self.missile_fired_count = obj.missile_fired_count
        self.missile_fire_fail_count = obj.missile_fire_fail_count

        self.home_returned_count = obj.home_returned_count  # 到基地次数

    def to_dict(self):
        return {
            **super().to_dict(),
            'collision_radius'           : self.collision_radius,
            'missile_count'              : self.missile_count,
            'fuel'                       : self.fuel,
            'fuel_consumption_rate'      : self.fuel_consumption_rate,
            'radar_radius'               : self.radar_radius,
            'last_fire_missile_time'     : self.last_fire_missile_time,
            'missile_hit_enemy'          : self.missile_hit_enemy,
            'missile_hit_enemy_count'    : self.missile_hit_enemy_count,
            'missile_hit_self'           : self.missile_hit_self,
            'missile_hit_self_count'     : self.missile_hit_self_count,
            'missile_miss'               : self.missile_miss,
            'missile_miss_count'         : self.missile_miss_count,
            'missile_evade_success'      : self.missile_evade_success,
            'missile_evade_success_count': self.missile_evade_success_count,
            'missile_fired'              : self.missile_fired,
            'missile_fired_count'        : self.missile_fired_count,
            'missile_fire_fail_count'    : self.missile_fire_fail_count,
            'aircraft_collided_count'    : self.aircraft_collided_count,
            'home_returned_count'        : self.home_returned_count,
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
                   position=self.waypoint.location,
                   img_path=f'aircraft_{self.color}.svg',
                   label=self.name,
                   rotate=self.waypoint.psi + 180)
        if self.destroyed:
            render_img(options=self.options,
                       screen=screen,
                       img_path='explosion.svg',
                       position=self.waypoint.location)

        # 画出导航轨迹
        render_route(options=self.options, screen=screen, route=self.render_route, color=self.color)

        # 画出雷达圆圈
        render_circle(
                options=self.options,
                screen=screen,
                position=self.waypoint.location,
                radius=self.radar_radius,
                color='green'
        )

    def update(self, delta_time: float):
        area = self.area
        assert area is not None, f'Cannot update'

        while not self.waiting_actions.empty():
            # 执行动作
            action = self.waiting_actions.get_nowait()
            if self.consumed_actions.full():
                self.consumed_actions.get_nowait()
            self.consumed_actions.put_nowait((round(area.time), action))  # 将消费的动作放入最近消费动作序列中

            action_type = int(action[0])
            action_target = action[1:]
            # print(f'{self.name} 执行动作', Actions(action_type).name)
            if action_type == Actions.go_to_location:
                self.go_to_location(target=action_target)
            elif action_type == Actions.fire_missile:
                # 寻找离目标点最近的飞机
                self.fire_missile(target=action_target)
            elif action_type == Actions.go_home:
                self.go_home()

        # 执行移动
        if not self.update_follow_route():
            self.update_move_forward(delta_time=delta_time)

        if self.options.destroy_on_boundary_exit and not self.is_in_game_range and self.last_is_in_game_range:
            self.destroy(reason=DestroyReason.OUT_OF_GAME_RANGE)

        self.last_is_in_game_range = self.is_in_game_range

        # 记忆路径点
        self.position_memory.add_position(self.waypoint.location)

        if self.fuel <= 0:
            return

        # 消耗汽油
        self.fuel -= self.fuel_consumption_rate * delta_time
        # 检查剩余油量
        if self.fuel <= 0:
            # 只会在油量耗尽的那一刻触发一次被摧毁
            # 耗尽燃油
            self.destroy(reason=DestroyReason.FUEL_DEPLETION)
            self.fuel_depletion_count += 1

    def go_home(self):
        home_position = self.area.get_home(self.color).waypoint.location
        self.go_to_location(target=home_position)

    def fire_missile(self, target: tuple[float, float]):
        """
        朝目标点发射导弹
        Args:
            target: 目标点

        Returns:

        """
        if not self.can_fire_missile():
            self.missile_fire_fail_count += 1
            return

        self.last_fire_missile_time = self.area.time

        # 寻找离目标点最近的飞机
        min_dis = float('inf')
        fire_enemy: Aircraft | None = None
        for enemy in self.area.objs.values():
            if isinstance(enemy, Aircraft) and enemy.color != self.color:
                dis = enemy.distance(target)
                if dis < min_dis and self.distance(enemy) <= self.radar_radius:
                    # 只能朝雷达范围内的飞机发射导弹
                    min_dis = dis
                    fire_enemy = enemy

        if fire_enemy is not None:
            self.missile_count -= 1
            missile = Missile(
                    name=f'{self.name}_missile_{self.missile_fired_count}',
                    source=self,
                    target=fire_enemy,
                    time=self.area.time)
            self.missile_fired.append(missile.name)
            self.missile_fired_count = len(self.missile_fired)

            self.area.add_obj(missile)
            # print(f'发射了导弹: {missile.name}')

    def predict_missile_intercept_point(self, target_wpt: Waypoint, target_speed: float) -> InterceptPointResult | None:
        """
        预测自己发射的导弹拦截对方的目标点
        :param target_wpt: 目标航迹点
        :param target_speed: 目标的移动速度
        :return:
        """
        return optimal_predict_intercept_point(
                self_wpt=self.waypoint,
                self_speed=self.options.missile_speed,
                self_turn_radius=self.options.missile_min_turn_radius,
                target_wpt=target_wpt,
                target_speed=target_speed,
        )

    def on_collision(self, obj: WorldObj):
        if isinstance(obj, Aircraft):
            self.destroy(reason=DestroyReason.COLLIDED_WITH_AIRCRAFT, source=obj.name)
            self.aircraft_collided_count += 1
        elif isinstance(obj, Missile) and obj.source.name != self.name:
            if self.options.missile_can_only_hit_enemy and obj.color == self.color:
                # 导弹只能攻击敌人
                return
            # 被导弹命中了
            self.destroy(reason=DestroyReason.COLLIDED_WITH_MISSILE, source=obj.name)
            self.on_missile_hit_self(missile=obj)

    def in_radar_range(self, obj: WorldObj | Waypoint | tuple[float, float] | np.ndarray) -> bool:
        """
        obj是否在雷达范围内
        :param obj:
        :return:
        """
        return self.distance(obj) <= self.radar_radius

    def on_missile_hit_enemy(self, missile: Missile, enemy: Aircraft):
        """自己发出的导弹命中敌人"""
        self.missile_hit_enemy.append((missile.name, enemy.name))
        self.missile_hit_enemy_count += 1

    def on_missile_hit_self(self, missile: Missile):
        """自己被别人发出的导弹命中"""
        self.missile_hit_self.append(missile.name)
        self.missile_hit_self_count += 1

    def on_missile_miss(self, missile: Missile):
        self.missile_miss.append(missile.name)
        self.missile_miss_count += 1
        # 导弹针对的飞机规避成功
        missile.target.missile_evade_success_count += 1
        missile.target.missile_evade_success.append(missile.name)

    def on_return_home(self):
        """
        回到基地，在一定时间间隔内只会触发一次
        options.home_return_time_interval 返回基地的时间间隔
        """
        self.home_returned_count += 1
        # 加油
        if self.options.home_refuel and self.fuel < self.options.home_refuel_threshold_capacity:
            self.fuel = self.options.aircraft_fuel_capacity
        # 补充导弹
        if self.options.home_replenish_missile and self.missile_count < self.options.home_replenish_missile_threshold_count:
            self.missile_count = self.options.aircraft_missile_count

    def can_fire_missile(self) -> bool:
        """是否可以发射导弹"""
        if self.missile_count <= 0:
            return False

        if (
                self.area.time - self.last_fire_missile_time) < self.options.aircraft_fire_missile_interval:
            return False

        enemy = self.area.find_nearest_enemy(
                agent_name=self.name,
                ignore_radar=False)

        if enemy is None:
            # self.put_update_message('No nearest enemy')
            return False

        return True


class Missile(WorldObj):
    def __init__(self, name: str, source: Aircraft, target: Aircraft, time: float):
        """
        :param source:
        :param target:
        """
        super().__init__(
                type='missile',
                options=source.options,
                name=name,
                color=source.color,
                speed=source.options.missile_speed,
                turn_radius=source.options.missile_min_turn_radius,
                waypoint=source.waypoint,
                collision_radius=source.options.missile_collision_radius
        )
        self.turn_radius = self.options.missile_min_turn_radius
        self.fire_time = time  # 发射的时间
        self.source: Aircraft = source
        self.target: Aircraft = target
        self.fuel = self.options.missile_fuel_capacity
        self.fuel_consumption_rate = self.options.missile_fuel_consumption_rate

        self._last_generate_route_time = 0

    def copy_from(self, obj: Missile):
        super().copy_from(obj)
        self.collision_radius = obj.collision_radius
        self.turn_radius = obj.turn_radius
        self.fire_time = obj.fire_time  # 发射的时间
        self.source: Aircraft = obj.source
        self.target: Aircraft = obj.target
        self.fuel = obj.fuel
        self.fuel_consumption_rate = obj.fuel_consumption_rate
        self._last_generate_route_time = obj._last_generate_route_time

    def render(self, screen):
        # print('render missile', self.name, self.screen_position, self.destroyed)
        if self.destroyed:
            return

        render_img(
                options=self.options,
                screen=screen,
                position=self.waypoint.location,
                img_path=f'missile_{self.color}.svg',
                # label=self.name,
                rotate=self.waypoint.psi + 90,
        )

        render_route(
                options=self.options,
                screen=screen,
                route=self.render_route,
                color=self.color,
                count=20
        )

    def destroy(self, reason: str, source=None):
        if self.destroyed:
            return
        super().destroy(reason=reason, source=source)
        if reason != DestroyReason.COLLIDED_WITH_AIRCRAFT:
            # 被摧毁的原因不是碰撞飞机，说明没有命中过敌机
            self.source.on_missile_miss(self)

    def update(self, delta_time: float):
        self.fuel -= self.fuel_consumption_rate * delta_time

        if self.fuel <= 0:
            # 燃油耗尽，说明没有命中过敌机
            self.destroy(reason=DestroyReason.FUEL_DEPLETION)

        # self.go_to_location(target=self.target.waypoint.location)
        if self.area.time - self._last_generate_route_time > self.options.missile_reroute_interval:
            self._last_generate_route_time = self.area.time

            self.go_to_location(target=self.target.waypoint.location)
        # # 每隔1秒重新生成一次轨迹
        # hit_param = calc_optimal_path(
        #         start=self.waypoint,
        #         target=self.target.waypoint.location,
        #         turn_radius=self.turn_radius
        # )
        # if hit_param.length != float('inf'):
        #     self.route = hit_param.build_route(step=delta_time * self.speed)  # 生成轨迹
        #     self.route_index = 0

        if not self.update_follow_route():
            self.update_move_forward(delta_time=delta_time)

    def on_collision(self, obj: WorldObj):
        if isinstance(obj, Aircraft):
            if obj.color == self.color:
                # 导弹暂时不攻击友方 TODO: 未来可能会修改
                return
            self.destroy(reason=DestroyReason.COLLIDED_WITH_AIRCRAFT, source=obj.name)
            self.source.on_missile_hit_enemy(missile=self, enemy=obj)


class Bullseye(WorldObj):
    def __init__(self, options: Options):
        super().__init__(type='bullseye', options=options, name='bullseye', color='white', waypoint=Waypoint())
        self.radius = options.bullseye_safe_radius()  # 安全半径

    def render(self, screen):
        # 渲染安全区域
        render_circle(options=self.options,
                      screen=screen,
                      radius=1,
                      position=self.waypoint.location,
                      color='black',
                      width=3)

        render_circle(options=self.options,
                      screen=screen,
                      radius=self.radius,
                      position=self.waypoint.location,
                      color='grey',
                      width=3)


class Home(WorldObj):
    def __init__(self, name: str, color: str, options: Options, waypoint: Waypoint):
        super().__init__(type='home', options=options, name=name, color=color, waypoint=waypoint)
        self.radius = options.home_area_radius
        self.in_range_objs = { }

    def render(self, screen):
        render_img(options=self.options,
                   screen=screen,
                   position=self.waypoint.location,
                   img_path=f'home_{self.color}.svg',
                   label=self.name)
        # 画出安全圆圈
        render_circle(options=self.options, screen=screen, position=self.waypoint.location, radius=self.radius,
                      color='green')

    def to_dict(self):
        return {
            **super().to_dict(),
            'radius'       : self.radius,
            'in_range_objs': self.in_range_objs
        }

    def update(self, delta_time: float):
        # 查看哪些飞机飞到了基地附近
        assert self.area is not None
        area = self.area

        for obj in area.objs.values():
            if not isinstance(obj, Aircraft):
                continue
            if self.distance(obj) > self.radius:
                # 不在基地范围内
                if obj.name in self.in_range_objs:
                    in_time = self.in_range_objs[obj.name]
                    del self.in_range_objs[obj.name]
                    self.on_aircraft_out(obj, in_time=in_time)
            else:
                # 在基地范围内
                if obj.name not in self.in_range_objs:
                    self.in_range_objs[obj.name] = area.time
                    self.on_aircraft_in(obj)
                elif area.time - self.in_range_objs[obj.name] >= self.options.home_return_time_interval:
                    self.on_aircraft_in(obj)
                    self.in_range_objs[obj.name] = area.time

    def on_aircraft_in(self, aircraft: Aircraft):
        # 飞机飞进来了
        if aircraft.color == self.color:
            aircraft.on_return_home()
        else:
            # 摧毁敌机
            if self.options.home_attack:
                aircraft.destroy(reason=DestroyReason.HOME_ATTACK, source=self.name)

    def on_aircraft_out(self, aircraft: Aircraft, in_time: float):
        """
        飞机飞出去了
        Args:
            aircraft: 飞出去的飞机
            in_time: 飞机飞进来的时间

        Returns:

        """
        pass

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
