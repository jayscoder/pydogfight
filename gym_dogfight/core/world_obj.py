from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
from gym_dogfight.core.constants import *
from gym_dogfight.core.actions import *
import random
from queue import Queue
from gym_dogfight.utils.rendering import pygame_load_img
from gym_dogfight.core.models import Waypoint
from gym_dogfight.algos.traj import calc_optimal_path, OptimalPathParam
from gym_dogfight.algos.intercept import predict_intercept_point
import math
from gym_dogfight.utils.rendering import *
import weakref

if TYPE_CHECKING:
    from gym_dogfight.core.options import Options
    from gym_dogfight.core.battle_area import BattleArea


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, name: str, options: Options, type: str, color: str = '', x: float = 0, y: float = 0,
                 psi: float | None = None,
                 speed: float = 0, turn_radius: float = 0):
        assert speed >= 0
        self.name = name
        self.options = options
        self.type = type
        self.color = color
        self.speed = speed
        self.turn_radius = turn_radius
        self.contains = None
        self.collision_radius = 0

        self.x = x
        self.y = y

        self.curve: np.ndarray | None = None
        self.curve_index: int = -1

        self.destroyed = False  # 是否已经被摧毁

        if psi is not None:
            self.psi = psi
        else:
            self.psi = random.randint(0, 359)

        self.actions = Queue()  # 等待消费的行为，每一项是个列表
        self._area = None
        # (0, -, -) 0代表什么也不做
        # （1, x, y）飞到指定位置
        # (2, x, y) 朝目标点发射导弹

    def put_action(self, action):
        if self.destroyed:
            return
        self.actions.put_nowait(action)

    def attach(self, battle_area: 'BattleArea'):
        self._area = weakref.ref(battle_area)

    @property
    def area(self) -> 'BattleArea' | None:
        if self._area is None:
            return None
        return self._area()

    @property
    def waypoint(self) -> Waypoint:
        return Waypoint(self.x, self.y, self.psi)

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def render(self, screen):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def distance(self, to: WorldObj) -> float:
        return ((self.x - to.x) ** 2 + (self.y - to.y) ** 2) ** 0.5

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

class Aircraft(WorldObj):

    def __init__(self,
                 name: str,
                 options: Options,
                 color: str,
                 x: float = 0,
                 y: float = 0,
                 psi: float | None = None,
                 ):
        super().__init__(name=name,
                         options=options,
                         type='aircraft',
                         color=color,
                         speed=options.aircraft_speed,
                         turn_radius=options.aircraft_min_turn_radius,
                         x=x,
                         y=y,
                         psi=psi)
        self.collision_radius = options.aircraft_missile_count
        self.missile_count = options.aircraft_missile_count  # 剩余的导弹数
        self.fuel = options.aircraft_fuel_capacity  # 飞机剩余油量
        self.fuel_consumption_rate = options.aircraft_fuel_consumption_rate
        self.radar_radius = options.aircraft_radar_radius
        self.missile_destroyed_enemies = []  # 导弹摧毁的敌机

    def render(self, screen):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise e

        assert screen is not None

        # 加载飞机图像（确保路径正确）
        aircraft_img = pygame_load_img(f'aircraft_{self.color}.svg').convert_alpha()
        explosion_img = pygame_load_img(f'explosion.svg').convert_alpha()

        # 可能需要根据飞机的当前位置和角度调整图像
        # 如果psi是角度，确保它是以度为单位
        aircraft_img = pygame.transform.rotate(aircraft_img, -self.psi)
        # aircraft_img = pygame.transform.smoothscale(aircraft_img, (32, 32))
        explosion_img = pygame.transform.smoothscale(explosion_img, (26, 26))

        # 获取飞机图像的矩形区域
        aircraft_img_rect = aircraft_img.get_rect()
        explosion_img_rect = explosion_img.get_rect()

        # 调整坐标系统：从游戏世界坐标转换为屏幕坐标
        screen_x, screen_y = game_point_to_screen_point(
                (self.x, self.y),
                game_size=self.options.game_size,
                screen_size=self.options.screen_size)

        # 设置飞机图像的位置
        aircraft_img_rect.center = (screen_x, screen_y)
        explosion_img_rect.center = (screen_x, screen_y)

        # 创建字体对象
        font = pygame.font.Font(None, 16)  # 使用默认字体，大小为36

        # 渲染文本到 Surface 对象
        text_surface = font.render(self.name, True, (0, 0, 0))

        # 获取文本区域的矩形
        text_rect = text_surface.get_rect()
        text_rect.center = [aircraft_img_rect.center[0], aircraft_img_rect.center[1] + 18]

        # 画出导航轨迹
        if self.curve is not None:
            for i in range(0, len(self.curve), 10):
                cur = self.curve[i][:2]
                pygame.draw.circle(screen, COLORS[self.color], game_point_to_screen_point(
                        game_point=cur,
                        game_size=self.options.game_size,
                        screen_size=self.options.screen_size
                ), 1)

        # 画出雷达圆圈
        if self.radar_radius > 0:
            pygame.draw.circle(screen, COLORS['green'], (screen_x, screen_y),
                               game_length_to_screen_length(
                                       self.radar_radius,
                                       game_size=self.options.game_size,
                                       screen_size=self.options.screen_size
                               ), 1)

        # 将飞机图像绘制到屏幕上
        screen.blit(aircraft_img, aircraft_img_rect)
        if self.destroyed:
            screen.blit(explosion_img, explosion_img_rect)
        screen.blit(text_surface, text_rect)

    def update(self, delta_time: float):
        if self.destroyed:
            return
        area = self.area
        if area is None:
            self.destroyed = True
            return

        if not self.actions.empty():
            # 先执行动作
            action = self.actions.get_nowait()
            action_type = int(action[0])
            if action_type == Actions.go_to_location:
                # 需要移动
                param = calc_optimal_path(self.waypoint, (action[1], action[2]), self.turn_radius)
                path = param.generate_traj(delta_time * self.speed)
                self.curve = path
                self.curve_index = 0
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
                    if area.render_mode == 'human':
                        area.add_obj(Missile(source=self, target=fire_enemy, time=area.duration))
                    if random.random() < self.options.predict_missile_hit_prob(self, fire_enemy):
                        fire_enemy.destroyed = True
                        self.missile_destroyed_enemies.append(fire_enemy.name)

        # 执行移动
        if self.curve is not None and self.curve.shape[0] > self.curve_index >= 0:
            self.x = self.curve[self.curve_index, 0]
            self.y = self.curve[self.curve_index, 1]
            self.psi = self.curve[self.curve_index, 2]
            self.curve_index += 1
        else:
            # 朝着psi的方向移动, psi是航向角，0度指向正北，90度指向正东
            # 将航向角从度转换为弧度
            x_theta = self.waypoint.standard_rad
            # 计算 x 和 y 方向上的速度分量
            dx = self.speed * math.cos(x_theta) * delta_time  # 正东方向为正值
            dy = self.speed * math.sin(x_theta) * delta_time  # 正北方向为正值

            # 更新 obj 的位置
            self.x += dx
            self.y += dy

        self.fuel -= self.fuel_consumption_rate * delta_time

        # 检查剩余油量
        if self.fuel <= 0:
            self.destroyed = True

        if self.options.destroy_on_boundary_exit:
            # 检查是否跑出了游戏范围
            game_x_range = (-self.options.game_size[0] / 2, self.options.game_size[0] / 2)
            game_y_range = (-self.options.game_size[1] / 2, self.options.game_size[1] / 2)

            if self.x < game_x_range[0] or self.x > game_x_range[1]:
                self.destroyed = True
            elif self.y < game_y_range[0] or self.y > game_y_range[1]:
                self.destroyed = True

    def predict_missile_intercept_point(self, enemy: Aircraft) -> tuple[float, float] | None:
        """
        预测自己发射的导弹拦截对方的目标点
        :param enemy:
        :return:
        """
        return predict_intercept_point(
                target=enemy.waypoint, target_speed=enemy.speed,
                self_speed=self.options.missile_speed,
                calc_optimal_dis=lambda p: calc_optimal_path(
                        start=self.waypoint,
                        target=(enemy.x, enemy.y),
                        turn_radius=self.options.missile_min_turn_radius
                ).length)

    def predict_aircraft_intercept_point(self, enemy: Aircraft) -> tuple[float, float] | None:
        """
        预测自己拦截敌方的目标点
        :param enemy:
        :return:
        """
        return predict_intercept_point(
                target=enemy.waypoint, target_speed=enemy.speed,
                self_speed=self.speed,
                calc_optimal_dis=lambda p: calc_optimal_path(
                        start=self.waypoint,
                        target=(enemy.x, enemy.y),
                        turn_radius=self.turn_radius
                ).length)


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
        )

        self.time = time  # 发射的时间
        from gym_dogfight.algos.traj import calc_optimal_path

        hit_param = calc_optimal_path(
                start=source.waypoint,
                target=(target.x, target.y),
                turn_radius=self.turn_radius
        )
        self.traj = hit_param.generate_traj(step=hit_param.length / 10)  # 生成十个点的轨迹

    def render(self, screen):
        if self.destroyed:
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise e

        if self.traj is not None:
            for point in self.traj:
                # 绘制轨迹
                screen_p = game_point_to_screen_point(
                        game_point=(point[0], point[1]),
                        game_size=self.options.game_size,
                        screen_size=self.options.screen_size
                )
                pygame.draw.circle(screen, COLORS[self.color], screen_p, 1)  # 绘制轨迹点

    def update(self, delta_time: float):
        # if area.duration - self.time > self.options.missile_render_duration:
        #     self.destroyed = True
        pass


class Home(WorldObj):
    def __init__(self, name: str, color: str, options: Options, x: float, y: float):
        super().__init__(type='home', options=options, name=name, color=color, x=x, y=y)
        self.radius = options.home_area_radius

    def render(self, screen):
        # 加载基地图像（确保路径正确）
        home_img = pygame_load_img(f'home_{self.color}.svg').convert_alpha()

        # 获取飞机图像的矩形区域
        home_img_rect = home_img.get_rect()

        # 调整坐标系统：从游戏世界坐标转换为屏幕坐标
        screen_x, screen_y = game_point_to_screen_point(
                (self.x, self.y),
                game_size=self.options.game_size,
                screen_size=self.options.screen_size)

        home_img_rect.center = (screen_x, screen_y)

        # 创建字体对象
        font = pygame.font.Font(None, 16)  # 使用默认字体，大小为36

        # 渲染文本到 Surface 对象
        text_surface = font.render(self.name, True, (0, 0, 0))

        # 获取文本区域的矩形
        text_rect = text_surface.get_rect()
        text_rect.center = [home_img_rect.center[0], home_img_rect.center[1] + 22]

        # 画出安全圆圈
        if self.radius > 0:
            pygame.draw.circle(screen, COLORS['green'], (screen_x, screen_y),
                               game_length_to_screen_length(
                                       self.radius,
                                       game_size=self.options.game_size,
                                       screen_size=self.options.screen_size
                               ), 1)

        # 将飞机图像绘制到屏幕上
        screen.blit(home_img, home_img_rect)
        screen.blit(text_surface, text_rect)

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
