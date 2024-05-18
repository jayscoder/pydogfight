import numpy as np

from pydogfight.core.world_obj import *
from pydogfight.core.battle_area import BattleArea
from gymnasium.spaces.space import Space
import gymnasium as gym


class ObsUtils:
    """
    观测工具类
    """

    W = 7  # 观测到的矩阵的宽度

    WATCH_MISSILES = 10  # 最多观测10个导弹

    def __init__(self, battle_area: BattleArea, agent_name: str):
        """
        Args:
            battle_area:
        """
        self.options = battle_area.options
        self.battle_area = battle_area
        self.N = len(battle_area.options.agents()) + self.WATCH_MISSILES  # 最多同时记录所有飞机、5个导弹的信息
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(self.N, self.W), dtype=np.float32)
        self.cache = { }
        self.agent_name = agent_name

        if agent_name in battle_area.options.red_agents or agent_name == battle_area.options.red_home:
            self.agent_color = 'red'
        else:
            self.agent_color = 'blue'

    def reset(self):
        self.cache.clear()

    def empty_obs(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    @classmethod
    def empty_obs_line(cls):
        return np.zeros((cls.W,), dtype=np.float32)

    def gen_self_obs_line(self, agent: WorldObj):
        """获取自己的观测"""
        obs = self.empty_obs_line()

        obs[0] = OBJECT_TO_IDX[agent.type]
        obs[1] = 0

        if isinstance(agent, Aircraft):
            obs[2] = int(agent.can_fire_missile())
            obs[3] = agent.missile_count / self.options.aircraft_missile_count
        # obs[4] = agent.speed / agent.radar_radius
        # obs[5] = int(agent.destroyed)

        # if agent.route_param is not None:
        #     # 自己当前设定的目标点
        #     rel_pt = agent.waypoint.relative_polar_waypoint(other=agent.route_param.target)
        #     obs[2] = rel_pt.r / agent.radar_radius
        #     obs[3] = np.deg2rad(rel_pt.theta)
        #     obs[4] = np.deg2rad(rel_pt.phi)
        return obs

    def gen_aircraft_obs_line(self, agent: WorldObj, obj: Aircraft):
        """
        Args:
            agent:
            obj:
        Returns:

        """
        if agent.name == obj.name:
            return self.gen_self_obs_line(agent=agent)
        rel_pt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
        obs = self.empty_obs_line()
        obs[0] = OBJECT_TO_IDX[obj.type]
        obs[1] = int(obj.color != agent.color)

        if obj.destroyed:
            return obs
        obs[2] = int(obj.destroyed)
        if obj.color != agent.color and not agent.options.obs_ignore_radar:
            if not self.battle_area.can_view_obj(self_color=agent.color, obj=obj):
                # 判断自己这边能不能看到对方
                return obs

        obs[3] = rel_pt.r / max(self.options.game_size)
        obs[4] = np.deg2rad(rel_pt.theta)
        obs[5] = np.deg2rad(rel_pt.phi)
        obs[6] = int(obj.can_fire_missile())

        # obs[6] = 0  # 不知道敌机的剩余导弹数量
        # obs[7] = obj.speed / agent.radar_radius

        # obj.fuel / agent.options.aircraft_fuel_capacity,  # 8
        # obj.radar_radius / agent.radar_radius,  # 9
        # obj.missile_count / agent.options.aircraft_missile_count,  # 10

        # if obj.color != agent.color:
        #     if agent.options.obs_ignore_enemy_fuel:
        #         # 不知道敌机的油量和导弹数
        #         obs[8] = -1
        #     if agent.options.obs_ignore_enemy_missile_count:
        #         # 不知道敌机的剩余导弹数
        #         obs[10] = -1
        return obs

    def gen_missile_obs_line(self, agent: WorldObj, obj: Missile):
        rel_pt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
        obs = self.empty_obs_line()
        obs[0] = OBJECT_TO_IDX[obj.type]
        obs[1] = int(obj.color != agent.color)

        if obj.destroyed:
            return

        obs[2] = int(obj.destroyed)
        if obj.color != agent.color and not agent.options.obs_ignore_radar:
            if not self.battle_area.can_view_obj(self_color=agent.color, obj=obj):
                # 判断自己这边能不能看到对方
                return

        obs[3] = rel_pt.r / max(self.options.game_size)
        obs[4] = np.deg2rad(rel_pt.theta)
        obs[5] = np.deg2rad(rel_pt.phi)

        # obs[5] = obj.fuel / agent.options.missile_fuel_capacity

        # obs[6] = obj.speed / agent.radar_radius
        # obs[7] = obj.turn_radius / agent.radar_radius
        #

        # if obj.color != agent.color:
        #     if agent.options.obs_ignore_enemy_missile_fuel:
        #         # 不知道敌方导弹的剩余油量
        #         obs[8] = -1
        return obs

    def gen_obs(self):
        """
        获取agent视角的obs
        注意这里的坐标用相对极坐标来表示
        :return: np.ndarray
        """
        agent = self.battle_area.get_obj(self.agent_name)
        index = 0
        obs = np.zeros(self.observation_space.shape)
        if isinstance(agent, Aircraft):
            obs[index, :] = self.gen_self_obs_line(agent)
            index += 1

        # 飞机
        if agent.color == 'red':
            self_aircrafts = [self.battle_area.get_agent(agent_name) for agent_name in self.options.red_agents]
            enemy_aircrafts = [self.battle_area.get_agent(agent_name) for agent_name in self.options.blue_agents]
            self_missiles = self.battle_area.get_missiles(color='red')
            enemy_missiles = self.battle_area.get_missiles(color='blue')
        else:
            self_aircrafts = [self.battle_area.get_agent(agent_name) for agent_name in self.options.blue_agents]
            enemy_aircrafts = [self.battle_area.get_agent(agent_name) for agent_name in self.options.red_agents]
            self_missiles = self.battle_area.get_missiles(color='blue')
            enemy_missiles = self.battle_area.get_missiles(color='red')

        for obj in self_aircrafts:
            if obj.name == agent.name:
                continue
            obs[index, :] = self.gen_aircraft_obs_line(agent=agent, obj=obj)
            index += 1

        for obj in enemy_aircrafts:
            obs[index, :] = self.gen_aircraft_obs_line(agent=agent, obj=obj)
            index += 1

        for obj in enemy_missiles:
            if index >= len(obs):
                break
            if obj.destroyed:
                continue
            obs[index, :] = self.gen_missile_obs_line(agent=agent, obj=obj)
            index += 1

        for obj in self_missiles:
            if index >= len(obs):
                break
            if obj.destroyed:
                continue
            obs[index, :] = self.gen_missile_obs_line(agent=agent, obj=obj)
            index += 1
        return obs

    # # 基地默认是知道的（不考虑雷达）
    # for obj in self.battle_area.homes:
    #     obs[index, :] = self.gen_home_obs(agent=agent, obj=obj)
    #     index += 1

    # # 牛眼
    # obs[index, :] = self.gen_bullseye_obs(agent=agent, obj=self.battle_area.bullseye)
    # index += 1

    # def gen_bullseye_obs(self, agent: Aircraft, obj: Bullseye):
    #     rel_pt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
    #     obs = self.empty_obs_line()
    #     obs[0] = OBJECT_TO_IDX[obj.type]
    #     obs[2] = int(obj.destroyed)
    #     obs[3] = rel_pt.r / agent.radar_radius
    #     obs[4] = np.deg2rad(rel_pt.theta)
    #     return obs

    # @classmethod
    # def gen_home_obs(cls, agent: Aircraft, obj: Home):
    #     rel_pt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
    #     obs = cls.empty_obs_line()
    #     obs[0] = OBJECT_TO_IDX[obj.type]
    #     obs[1] = int(obj.color != agent.color)
    #     obs[2] = int(obj.destroyed)
    #     obs[3] = rel_pt.r / agent.radar_radius
    #     obs[4] = np.deg2rad(rel_pt.theta)
    #     return obs

    # def gen_global_obs(self):
    #     if self.options.self_side == 'red':
    #         return self.gen_agent_obs(self.options.red_agents[0])
    #     else:
    #         return self.gen_agent_obs(self.options.blue_agents[0])
    #     obs = np.zeros(self.observation_space.shape)
    #     i = 0
    #     # 飞机
    #     for obj in self.battle_area.agents:
    #         obs[i, :] = [
    #             OBJECT_TO_IDX[obj.type],  # 0
    #             COLOR_TO_IDX[obj.color],  # 1
    #             int(obj.destroyed),  # 2
    #             obj.waypoint.x,  # 3
    #             obj.waypoint.y,  # 4
    #             obj.waypoint.psi,  # 5
    #             obj.speed,  # 6
    #             obj.turn_radius,  # 7
    #             obj.fuel,  # 8
    #             obj.radar_radius,  # 9
    #             obj.missile_count,  # 10
    #         ]
    #         i += 1
    #
    #     # 基地
    #     for obj in self.battle_area.homes:
    #         obs[i, :] = [
    #             OBJECT_TO_IDX[obj.type],  # 0
    #             COLOR_TO_IDX[obj.color],  # 1
    #             int(obj.destroyed),  # 2
    #             obj.waypoint.x,  # 3
    #             obj.waypoint.y,  # 4
    #             0,  # 5
    #             0,  # 6
    #             0,  # 7
    #             0,  # 8
    #             0,  # 9
    #             0,  # 10
    #         ]
    #         i += 1
    #
    #     # 导弹
    #     for obj in self.battle_area.missiles:
    #         if obj.destroyed:
    #             continue
    #         if i >= len(obs):
    #             break
    #         obs[i, :] = [
    #             OBJECT_TO_IDX[obj.type],  # 0
    #             COLOR_TO_IDX[obj.color],  # 1
    #             int(obj.destroyed),  # 2
    #             obj.waypoint.x,  # 3
    #             obj.waypoint.y,  # 4
    #             obj.waypoint.psi,  # 5
    #             obj.speed,  # 6
    #             obj.turn_radius,  # 7
    #             obj.fuel,  # 8
    #             0,  # 9
    #             0,  # 10
    #         ]
    #         i += 1
    #
    #     return obs
