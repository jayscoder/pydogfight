import math
import random
import typing

import numpy as np

from pydogfight.utils.models import *


class Options:
    title: 'str' = 'Dogfight'

    debug: bool = True
    render: bool = False
    ### 实体设置 ###
    red_agents = ['red']
    blue_agents = ['blue']

    # def agents(self) -> list[str]:
    #     return self.blue_agents + self.red_agents
    def agents(self) -> list[str]:
        return self.red_agents + self.blue_agents

    red_home = 'red_home'
    blue_home = 'blue_home'
    bullseye = 'bullseye'

    self_side = 'red'  # 自己所处的战队（奖励会根据这个来给）

    ### 场景 ###
    max_duration = 60 * 60  # 一局对战最多时长60分钟，超过这个就会truncated
    screen_size = (800, 800)  # 屏幕宽度 屏幕高度
    render_fps = 50  # 渲染的fps
    delta_time = 0.1  # 每次env的更新步长
    update_interval = 1  # 每轮策略更新的时间间隔
    simulation_rate = 100.0  # 仿真的时间倍数，真实世界的1s对应游戏世界的多长时间

    reach_location_threshold = 2  # 用来判断是否接近目标点的时间片尺度（乘以delta_time*速度后就能得出距离多近就算到达目标点）

    def reach_location_interval(self):
        """判断是否接近目标点的时间长度"""
        return self.delta_time * self.reach_location_threshold

    obs_ignore_radar = False  # 是否忽略雷达（设置为true的话，生成单机观测时不会观测到雷达范围以内的敌机）
    obs_ignore_enemy_missile_fuel = False  # 是否忽略敌方导弹剩余油量
    obs_ignore_enemy_fuel = False  # 是否忽略敌方的剩余油量
    obs_ignore_enemy_missile_count = False  # 是否忽略敌方的剩余导弹数
    obs_ignore_destroyed = True  # 忽略掉被摧毁的实体

    obs_allow_memory = True  # 是否允许记忆敌机最近一次出现的位置

    indestructible = False  # 是否开启无敌模式，飞机不可被摧毁

    ### 常量 ###
    g = 9.8  # 重力加速度 m/s

    ### 战场 ###
    game_size = (5e4, 5e4)  # 战场宽度 50km 战场高度 50km
    destroy_on_boundary_exit = True  # 飞出战场边界是否会摧毁飞机

    collision_scale = 1  # 碰撞半径的倍数，越大代表越容易碰撞

    ### 飞机 ###
    aircraft_missile_count: int = 10  # 飞机上装载的导弹数量
    aircraft_speed: float = 200  # 飞机飞行速度200m/s

    @property
    def aircraft_collision_radius(self):
        return max(15.0, self.aircraft_speed * self.delta_time * self.collision_scale)  # 飞机的碰撞半径，用来进行碰撞检查，设为0就不会检查碰撞了

    aircraft_fuel_consumption_rate: float = 1  # 飞机耗油速度，每秒消耗多少油
    aircraft_fuel_capacity: float = aircraft_fuel_consumption_rate * 1800  # 飞机载油量，在这里飞机最多能飞1800秒
    aircraft_fuel_bingo_fuel = aircraft_fuel_capacity / 5  # 飞机bingo油量，留20%

    aircraft_max_centripetal_acceleration = 9 * g  # 飞机最大向心加速度

    aircraft_min_turn_radius = aircraft_speed ** 2 / aircraft_max_centripetal_acceleration  # 飞机最小转弯半径558m
    # aircraft_min_turn_radius = 5000
    aircraft_predict_distance = aircraft_speed * 5  # 预测未来5秒的位置

    # aircraft_radar_radius = 1e4  # 雷达半径 10km
    aircraft_radar_radius = 15000  # 雷达半径 10km

    aircraft_fire_missile_interval = 15  # 发射导弹时间间隔

    aircraft_position_memory_sep = 10000  # 飞机记忆走过的路径点（用来提取未走过的敌方），以10000作为分隔点

    ### 导弹 ###
    # missile_max_threat_distance = 8e3  # 导弹最大威胁距离8km
    # missile_no_escape_distance = 2e3  # 导弹不可躲避距离2km

    missile_max_centripetal_acceleration = 20 * g  # 导弹最大向心加速度
    missile_speed = aircraft_speed * 5  # 导弹速度是飞机速度的5倍
    missile_min_turn_radius = missile_speed ** 2 / missile_max_centripetal_acceleration  # 导弹最小转弯半径 4023m

    @property
    def missile_collision_radius(self):
        return max(10.0, self.missile_speed * self.delta_time * self.collision_scale)

    # missile_collision_radius = max(15.0, missile_speed * delta_time * 5)  # 导弹的碰撞半径

    missile_fuel_consumption_rate = 1
    missile_fuel_capacity = missile_fuel_consumption_rate * 60  # 导弹只能飞60秒: 26640m

    def missile_flight_duration(self):
        """导弹飞行时长"""
        return self.missile_fuel_capacity / self.missile_fuel_consumption_rate

    missile_reroute_interval = 0.1  # 导弹重新规划路径时间间隔
    missile_fire_interval = 5  # 每隔5 s最多发射一枚导弹

    missile_can_only_hit_enemy: bool = True  # 导弹是否只能攻击敌方（如果设为False，则导弹可以打中友方）

    ### 基地 ###
    home_area_radius = 2e3  # 基地范围半径
    home_return_time_interval = 10  # 触发返回基地的时间间隔

    home_refuel = True  # 基地是否具备重新加油功能
    home_refuel_threshold_capacity = aircraft_fuel_bingo_fuel * 2  # 基地重新加油的阈值（飞机剩余油量小于这个才会重新加油）
    home_replenish_missile = True  # 基地是否具备补充导弹的功能
    home_replenish_missile_threshold_count = aircraft_missile_count / 5  # 基地补充导弹的阈值（飞机剩余导弹数量小于这个才会重新补充导弹）

    home_attack = False  # 基地是否具备攻击功能（进入基地范围的敌机会被自动毁灭）

    ### 训练 ###
    win_reward = 100  # 胜利奖励
    lose_reward = -100  # 失败奖励
    draw_reward = -100  # 平局奖励
    step_reward = -0.1  # 每步的惩罚
    device: str = 'cpu'  # 训练使用的设备 cpu/mps/cuda

    # missile_hit_enemy_reward = 100  # 导弹命中敌机的奖励
    # missile_hit_self_reward = -100  # 被导弹命中的奖励
    # missile_miss_reward = -10  # 导弹没有命中的奖励
    # fuel_depletion_reward = -100 # 燃油耗尽奖励

    status_reward = {  # 状态奖励
        'SUCCESS': 0,
        'RUNNING': 0,
        'FAILURE': -1,
        'INVALID': -0.5
    }

    ### 策略 ###
    safe_boundary_distance = aircraft_speed * 20  # 距离边界的安全距离

    def bullseye_safe_radius(self):
        """牛眼的安全半径"""
        return min(self.game_size) / 2 - self.safe_boundary_distance

    def safe_boundary(self):
        return BoundingBox.from_center(
                center=[0, 0], size=[self.game_size[0] - self.safe_boundary_distance * 2,
                                     self.game_size[1] - self.safe_boundary_distance * 2])

    def validate(self):
        """校验是否合法"""
        assert self.delta_time > 0
        assert self.red_home != ''
        assert self.blue_home != ''

    def to_dict(self):
        d = { }
        temp_dict = {
            **self.__class__.__dict__,
            **self.__dict__
        }
        for k in temp_dict:
            v = temp_dict[k]
            if k.startswith('__'):
                continue
            if isinstance(v, type):
                continue
            if isinstance(v, typing.Callable):
                continue
            d[k] = v
        return d

    def load_dict(self, d: dict):
        for k, v in d.items():
            try:
                setattr(self, k, v)
            except Exception as e:
                print(f'setting {k} to {v} is not valid: {e}')

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def generate_random_point(self) -> tuple[float, float]:
        x = (random.random() * self.game_size[0] - self.game_size[0] / 2) * 0.9
        y = (random.random() * self.game_size[1] - self.game_size[1] / 2) * 0.9
        return x, y

    def generate_home_init_position(self, color: str) -> Tuple[float, float]:
        """
        生成基地的坐标点，第一个是红方基地的，第二个是蓝方基地的
        :param color: red/blue
        :return: x, y
        """
        if color == 'red':
            boundbox = BoundingBox(
                    left_top=(-self.game_size[0] * 3 / 8, -self.game_size[1] / 8),
                    size=(self.game_size[0] / 4, self.game_size[1] / 4),
            )

            return boundbox.center
        else:
            boundbox = BoundingBox(
                    (self.game_size[0] / 8, -self.game_size[1] / 8),
                    (self.game_size[0] / 4, self.game_size[1] / 4),
            )
            return boundbox.center

    def generate_aircraft_init_waypoint(
            self, color: str,
            home_position: tuple[float, float]) -> \
            tuple[float, float, float]:
        """
        生成飞机初始航迹点
        :param color: 战队颜色 red/blue
        :param home_position: 基地位置
        :return: (x, y, psi)
        """
        # theta = random.random() * 2 * math.pi
        # r = random.random() * self.home_area_radius
        #
        # x = home_position[0] + r * math.cos(theta)
        # y = home_position[1] + r * math.sin(theta)
        # psi = 90 - math.degrees(theta)
        x = home_position[0]
        y = home_position[1]
        psi = standard_to_heading(math.degrees(math.atan2(-y, -x)))
        return np.array([x, y, psi])

    def calc_screen_length(self, game_length: float) -> float:
        max_screen_size = max(self.screen_size)
        max_game_size = max(self.game_size)
        return (game_length / max_game_size) * max_screen_size

    def calc_game_length(self, screen_length: float) -> float:
        max_screen_size = max(self.screen_size)
        max_game_size = max(self.game_size)
        return (screen_length / max_screen_size) * max_game_size


if __name__ == '__main__':
    pass
    # options = Options()
    # options.red_agents = []
    # # print(Options.red_agents)
    # # print(options.to_dict())
    # di = options.to_dict()
    # options.from_dict(di)
    # print(options.to_dict())
    # print(di['__dict__'])
    # print(Options.__dict__)
    # print(options.missile_collision_radius)
