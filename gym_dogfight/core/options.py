import math
from typing import Callable, TYPE_CHECKING, List, Tuple
import random
import json

if TYPE_CHECKING:
    pass


class Options:
    ### 实体设置 ###
    red_agents = ['red_1']
    blue_agents = ['blue_1']
    red_home = 'red_home'  # 如果设置为空字符串则不会设置基地
    blue_home = 'blue_home'

    self_side = 'red'  # 自己所处的战队（奖励会根据这个来给）

    ### 场景 ###
    max_duration = 60 * 30  # 一局对战最多时长30分钟，超过这个就会truncated
    screen_size = (800, 800)  # 屏幕宽度 屏幕高度
    delta_time = 1 / 50  # 更新间隔时间
    simulation_rate = 20.0  # 仿真的速率倍数，越大代表越快

    ### 常量 ###
    g = 9.8  # 重力加速度 m/s

    ### 战场 ###
    game_size = (5e4, 5e4)  # 战场宽度 50km 战场高度 50km
    destroy_on_boundary_exit = True  # 飞出战场边界是否会摧毁飞机

    ### 飞机 ###

    aircraft_collision_radius: float = 10  # 飞机的碰撞半径为10m，用来进行碰撞检查，设为0就不会检查碰撞了
    aircraft_missile_count: int = 100  # 飞机上装载的导弹数量
    aircraft_speed: float = 222  # 飞机飞行速度220m/s 800km/h

    aircraft_fuel_capacity: float = 3200  # 飞机载油量 3200L
    aircraft_fuel_consumption_rate: float = aircraft_fuel_capacity / 1800  # 飞机耗油速度，在这里飞机最多能飞1800秒
    aircraft_fuel_bingo_fuel = aircraft_fuel_capacity / 5  # 飞机bingo油量，留20%

    aircraft_max_centripetal_acceleration = 9 * g  # 飞机最大向心加速度

    aircraft_min_turn_radius = aircraft_speed ** 2 / aircraft_max_centripetal_acceleration  # 飞机最小转弯半径558m
    # aircraft_min_turn_radius = 5000
    aircraft_predict_distance = aircraft_speed * 5  # 预测未来1秒的位置

    aircraft_radar_radius = 10e3  # 雷达半径 10km

    ### 导弹 ###
    missile_max_threat_distance = 8e3  # 导弹最大威胁距离8km
    missile_no_escape_distance = 2e3  # 导弹不可躲避距离2km

    missile_max_centripetal_acceleration = 20 * g  # 导弹最大向心加速度
    missile_render_duration: float = simulation_rate * 3  # 导弹画3秒
    missile_speed = aircraft_speed * 5  # 导弹速度是飞机速度的5倍
    missile_min_turn_radius = missile_speed ** 2 / missile_max_centripetal_acceleration  # 导弹最小转弯半径 6286m

    ### 基地 ###
    home_area_radius = 2e3  # 基地范围半径
    home_refuel = True  # 基地是否具备重新加油功能
    home_refuel_threshold_capacity = aircraft_fuel_bingo_fuel * 2  # 基地重新加油的阈值（飞机剩余油量小于这个才会重新加油）
    home_replenish_missile = True  # 基地是否具备补充导弹的功能
    home_replenish_missile_threshold_count = aircraft_missile_count / 5  # 基地补充导弹的阈值（飞机剩余导弹数量小于这个才会重新补充导弹）

    home_attack = False  # 基地是否具备攻击功能（进入基地范围的敌机会被自动毁灭）

    ### 训练 ###
    win_reward = 1000
    lose_reward = -1000
    draw_reward = -500
    step_punish_reward = -1
    step_update_delta_time = 0

    def to_dict(self):
        return {
            'red_agents'                            : self.red_agents,
            'blue_agents'                           : self.blue_agents,
            'red_home'                              : self.red_home,
            'blue_home'                             : self.blue_home,
            'self_side'                             : self.self_side,

            'max_duration'                          : self.max_duration,
            'screen_size'                           : self.screen_size,
            'delta_time'                            : self.delta_time,
            'simulation_rate'                       : self.simulation_rate,

            'g'                                     : self.g,
            'game_size'                             : self.game_size,
            'destroy_on_boundary_exit'              : self.destroy_on_boundary_exit,

            'aircraft_collision_radius'             : self.aircraft_collision_radius,
            'aircraft_missile_count'                : self.aircraft_missile_count,
            'aircraft_speed'                        : self.aircraft_speed,
            'aircraft_fuel_capacity'                : self.aircraft_fuel_capacity,
            'aircraft_fuel_consumption_rate'        : self.aircraft_fuel_consumption_rate,
            'aircraft_fuel_bingo_fuel'              : self.aircraft_fuel_bingo_fuel,
            'aircraft_max_centripetal_acceleration' : self.aircraft_max_centripetal_acceleration,
            'aircraft_min_turn_radius'              : self.aircraft_min_turn_radius,
            'aircraft_predict_distance'             : self.aircraft_predict_distance,
            'aircraft_radar_radius'                 : self.aircraft_radar_radius,

            'missile_max_threat_distance'           : self.missile_max_threat_distance,
            'missile_no_escape_distance'            : self.missile_no_escape_distance,
            'missile_max_centripetal_acceleration'  : self.missile_max_centripetal_acceleration,
            'missile_render_duration'               : self.missile_render_duration,
            'missile_speed'                         : self.missile_speed,
            'missile_min_turn_radius'               : self.missile_min_turn_radius,

            'home_area_radius'                      : self.home_area_radius,
            'home_refuel'                           : self.home_refuel,
            'home_refuel_threshold_capacity'        : self.home_refuel_threshold_capacity,
            'home_replenish_missile'                : self.home_replenish_missile,
            'home_replenish_missile_threshold_count': self.home_replenish_missile_threshold_count,
            'home_attack'                           : self.home_attack,

            'win_reward'                            : self.win_reward,
            'lose_reward'                           : self.lose_reward,
            'draw_reward'                           : self.draw_reward,
            'step_punish_reward'                    : self.step_punish_reward,
            'step_update_delta_time'                : self.step_update_delta_time
        }

    def from_dict(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    @property
    def agents(self) -> list[str]:
        return self.red_agents + self.blue_agents

    from gym_dogfight.core.world_obj import Aircraft

    def predict_missile_hit_prob(self, source: Aircraft, target: Aircraft):
        """
        预测导弹命中目标概率
        根据距离来判断敌方被摧毁的概率，距离越远，被摧毁的概率越低（基于MISSILE_MAX_THREAT_DISTANCE和MISSILE_NO_ESCAPE_DISTANCE）
        :param source: 发射导弹方
        :param target: 被导弹攻击方
        :return:
        """
        # 计算导弹发射轨迹
        from gym_dogfight.algos.traj import calc_optimal_path
        hit_point = source.predict_missile_intercept_point(enemy=target)

        if hit_point is None:
            return 0

        param = calc_optimal_path(
                start=source.waypoint,
                target=(target.x, target.y),
                turn_radius=self.missile_min_turn_radius
        )

        # 如果距离小于等于不可躲避距离，目标必定被摧毁
        if param.length <= self.missile_no_escape_distance:
            return 1

        # 如果距离超出最大威胁距离，目标不会被摧毁
        if param.length > self.missile_max_threat_distance:
            return 0

        # 在不可躲避距离和最大威胁距离之间，摧毁的概率随距离增加而减少
        hit_prob = (self.missile_max_threat_distance - param.length) / (
                self.missile_max_threat_distance - self.missile_no_escape_distance)

        return hit_prob

    def generate_random_point(self) -> tuple[float, float]:
        x = (random.random() * self.game_size[0] - self.game_size[0] / 2) * 0.9
        y = (random.random() * self.game_size[1] - self.game_size[1] / 2) * 0.9
        return x, y

    def generate_home_init_position(self) -> List[Tuple[float, float]]:
        """
        生成基地的坐标点，第一个是红方基地的，第二个是蓝方基地的
        :return: [red, blue]
        """
        from gym_dogfight.utils.common import generate_random_point

        red_home = generate_random_point(
                (-self.game_size[0] * 3 / 8, -self.game_size[1] / 8),
                (self.game_size[0] / 4, self.game_size[1] / 4),
        )
        blue_home = generate_random_point(
                (self.game_size[0] / 8, -self.game_size[1] / 8),
                (self.game_size[0] / 4, self.game_size[1] / 4),
        )
        return [red_home, blue_home]

    def generate_aircraft_init_waypoint(self, home_position: tuple[float, float], home_radius: float) -> tuple[
        float, float, float]:
        """
        生成飞机初始航迹点
        :param home_position: 基地位置
        :param home_radius: 基地辐射半径
        :return: (x, y, psi)
        """
        theta = random.random() * 2 * math.pi
        r = random.random() * home_radius

        x = home_position[0] + r * math.cos(theta)
        y = home_position[1] + r * math.sin(theta)
        psi = 90 - math.degrees(theta)

        return x, y, psi


if __name__ == '__main__':
    import jsonpickle
    options = Options()
    # setattr(options, 'agents', 1)
    # print(options.agents)
    print(getattr(options, 'agents', None))
    # for k in dir(options):
    #     print(k, type(getattr(options, k)))
