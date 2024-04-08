import random
from pydogfight.core.models import *


class Options:
    ### 实体设置 ###
    red_agents = ['red_1']
    blue_agents = ['blue_1']
    red_home = 'red_home'
    blue_home = 'blue_home'
    bullseye = 'bullseye'

    self_side = 'red'  # 自己所处的战队（奖励会根据这个来给）

    ### 场景 ###
    max_duration = 60 * 30  # 一局对战最多时长30分钟，超过这个就会truncated
    screen_size = (800, 800)  # 屏幕宽度 屏幕高度
    render_fps = 50  # 渲染的fps
    delta_time = 0.01  # 更新步长
    update_interval = 1  # 每次env更新的时间间隔
    simulation_rate = 100.0  # 仿真的速率倍数，越大代表越快，update_interval内更新几次（仅在render模式下生效）
    policy_interval = 0.1  # 每次策略的处理间隔时长
    reach_location_threshold = 2  # 用来判断是否接近目标点的时间片尺度（乘以policy_interval*速度后就能得出距离多近就算到达目标点）

    obs_ignore_radar = False  # 是否忽略雷达（设置为true的话，生成单机观测时不会观测到雷达范围以内的敌机）
    obs_ignore_enemy_missile_fuel = False  # 是否忽略敌方导弹剩余油量
    obs_ignore_enemy_fuel = False  # 是否忽略敌方的剩余油量
    obs_ignore_enemy_missile_count = False  # 是否忽略敌方的剩余导弹数
    obs_ignore_destroyed = True  # 忽略掉被摧毁的实体

    obs_allow_memory = True  # 是否允许记忆

    ### 常量 ###
    g = 9.8  # 重力加速度 m/s

    ### 战场 ###
    game_size = (5e4, 5e4)  # 战场宽度 50km 战场高度 50km
    destroy_on_boundary_exit = True  # 飞出战场边界是否会摧毁飞机

    ### 飞机 ###
    aircraft_missile_count: int = 2  # 飞机上装载的导弹数量
    aircraft_speed: float = 200  # 飞机飞行速度200m/s
    aircraft_collision_radius: float = max(10.0, aircraft_speed * delta_time * 1)  # 飞机的碰撞半径，用来进行碰撞检查，设为0就不会检查碰撞了

    aircraft_fuel_consumption_rate: float = 1  # 飞机耗油速度，每秒消耗多少油
    aircraft_fuel_capacity: float = aircraft_fuel_consumption_rate * 1800  # 飞机载油量，在这里飞机最多能飞1800秒
    aircraft_fuel_bingo_fuel = aircraft_fuel_capacity / 5  # 飞机bingo油量，留20%

    aircraft_max_centripetal_acceleration = 9 * g  # 飞机最大向心加速度

    aircraft_min_turn_radius = aircraft_speed ** 2 / aircraft_max_centripetal_acceleration  # 飞机最小转弯半径558m
    # aircraft_min_turn_radius = 5000
    aircraft_predict_distance = aircraft_speed * 5  # 预测未来5秒的位置

    # aircraft_radar_radius = 1e4  # 雷达半径 10km
    aircraft_radar_radius = 3e4  # 雷达半径 30km

    ### 导弹 ###
    # missile_max_threat_distance = 8e3  # 导弹最大威胁距离8km
    # missile_no_escape_distance = 2e3  # 导弹不可躲避距离2km

    missile_max_centripetal_acceleration = 20 * g  # 导弹最大向心加速度
    missile_speed = aircraft_speed * 5  # 导弹速度是飞机速度的5倍
    missile_min_turn_radius = missile_speed ** 2 / missile_max_centripetal_acceleration  # 导弹最小转弯半径 4023m

    missile_collision_radius = max(10.0, missile_speed * delta_time * 1)  # 导弹的碰撞半径

    missile_fuel_consumption_rate = 1
    missile_fuel_capacity = missile_fuel_consumption_rate * 30  # 导弹只能飞20秒: 26640m
    missile_reroute_interval = 1  # 导弹重新规划路径时间间隔
    missile_fire_interval = 5 # 每隔5 s最多发射一枚导弹

    ### 基地 ###
    home_area_radius = 2e3  # 基地范围半径
    home_refuel = True  # 基地是否具备重新加油功能
    home_refuel_threshold_capacity = aircraft_fuel_bingo_fuel * 2  # 基地重新加油的阈值（飞机剩余油量小于这个才会重新加油）
    home_replenish_missile = True  # 基地是否具备补充导弹的功能
    home_replenish_missile_threshold_count = aircraft_missile_count / 5  # 基地补充导弹的阈值（飞机剩余导弹数量小于这个才会重新补充导弹）

    home_attack = False  # 基地是否具备攻击功能（进入基地范围的敌机会被自动毁灭）

    ### 训练 ###
    win_reward = 3000
    lose_reward = -1000
    draw_reward = -500  #
    time_punish_reward = -1  # 时间惩罚（每s惩罚多少分）
    train = True  # 是否是训练模式

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
        assert self.policy_interval >= self.delta_time
        assert self.red_home != ''
        assert self.blue_home != ''

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
            'render_fps'                            : self.render_fps,
            'update_interval'                       : self.update_interval,

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

            # 'missile_max_threat_distance'           : self.missile_max_threat_distance,
            # 'missile_no_escape_distance'            : self.missile_no_escape_distance,
            'missile_max_centripetal_acceleration'  : self.missile_max_centripetal_acceleration,
            'missile_speed'                         : self.missile_speed,
            'missile_min_turn_radius'               : self.missile_min_turn_radius,
            'missile_collision_radius'              : self.missile_collision_radius,
            'missile_fuel_consumption_rate'         : self.missile_fuel_consumption_rate,
            'missile_fuel_capacity'                 : self.missile_fuel_capacity,
            'missile_reroute_interval'              : self.missile_reroute_interval,

            'home_area_radius'                      : self.home_area_radius,
            'home_refuel'                           : self.home_refuel,
            'home_refuel_threshold_capacity'        : self.home_refuel_threshold_capacity,
            'home_replenish_missile'                : self.home_replenish_missile,
            'home_replenish_missile_threshold_count': self.home_replenish_missile_threshold_count,
            'home_attack'                           : self.home_attack,

            'win_reward'                            : self.win_reward,
            'lose_reward'                           : self.lose_reward,
            'draw_reward'                           : self.draw_reward,
            'step_punish_reward'                    : self.time_punish_reward,
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

    def generate_random_point(self) -> tuple[float, float]:
        x = (random.random() * self.game_size[0] - self.game_size[0] / 2) * 0.9
        y = (random.random() * self.game_size[1] - self.game_size[1] / 2) * 0.9
        return x, y

    def generate_home_init_position(self) -> List[Tuple[float, float]]:
        """
        生成基地的坐标点，第一个是红方基地的，第二个是蓝方基地的
        :return: [red, blue]
        """
        from pydogfight.utils.common import generate_random_point

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

    def calc_screen_length(self, game_length: float) -> float:
        max_screen_size = max(self.screen_size)
        max_game_size = max(self.game_size)
        return (game_length / max_game_size) * max_screen_size

    def calc_game_length(self, screen_length: float) -> float:
        max_screen_size = max(self.screen_size)
        max_game_size = max(self.game_size)
        return (screen_length / max_screen_size) * max_game_size


if __name__ == '__main__':
    options = Options()
    print(options.aircraft_collision_radius)
    print(options.missile_collision_radius)
