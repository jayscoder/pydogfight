import numpy as np

from pydogfight.core.world_obj import *
from pydogfight.core.battle_area import BattleArea
from gymnasium.spaces.space import Space
import gymnasium as gym


class ObsUtils:
    """
    观测工具类
    """

    W = 11  # 观测到的矩阵的宽度

    def __init__(self, battle_area: BattleArea):
        """

        Args:
            battle_area:
        """
        self.battle_area = battle_area
        self.N = len(battle_area.options.agents()) + 3 + 7  # 最多同时记录所有飞机、基地、牛眼和7个导弹的信息
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(self.N, self.W), dtype=np.float32)
        self.cache = { }

    def reset(self):
        self.cache.clear()

    def empty_obs(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    @classmethod
    def gen_self_obs(cls, agent: Aircraft):
        """获取自己的观测"""
        return np.array([
            OBJECT_TO_IDX[agent.type],  # 0
            0,  # is_enemy 1
            int(agent.destroyed),  # 2
            0,  # r 3
            0,  # theta 4
            0,  # psi 5
            agent.speed / agent.radar_radius,  # 6
            agent.turn_radius / agent.radar_radius,  # 7
            agent.fuel / agent.options.aircraft_fuel_capacity,  # 8
            agent.radar_radius / agent.radar_radius,  # 9 radius
            agent.missile_count / agent.options.aircraft_missile_count,  # 10
        ])

    @classmethod
    def gen_aircraft_obs(cls, agent: Aircraft, obj: Aircraft, is_memory: bool):
        """

        Args:
            agent:
            obj:
            is_memory:

        Returns:

        """
        if agent.name == obj.name:
            return cls.gen_self_obs(agent=agent)
        rel_pt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
        obs = [
            OBJECT_TO_IDX[obj.type],  # 0
            int(obj.color != agent.color),  # 1
            0,  # is_destroyed/is_memory 2
            rel_pt.r / agent.radar_radius,  # r 3
            np.deg2rad(rel_pt.theta),  # theta 4
            np.deg2rad(rel_pt.phi),  # psi 5
            obj.speed / agent.radar_radius,  # 6
            obj.turn_radius / agent.radar_radius,  # 7
            obj.fuel / agent.options.aircraft_fuel_capacity,  # 8
            obj.radar_radius / agent.radar_radius,  # 9
            obj.missile_count / agent.options.aircraft_missile_count,  # 10
        ]

        if obj.destroyed:
            obs[2] = 1
        elif is_memory:
            obs[2] = -1

        if obj.color != agent.color:
            if agent.options.obs_ignore_enemy_fuel:
                # 不知道敌机的油量和导弹数
                obs[8] = -1
            if agent.options.obs_ignore_enemy_missile_count:
                # 不知道敌机的剩余导弹数
                obs[10] = -1
        return np.array(obs)

    @classmethod
    def gen_home_obs(cls, agent: Aircraft, obj: Home):
        rel_pt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
        obs = [
            OBJECT_TO_IDX[obj.type],  # 0
            int(obj.color != agent.color),  # 1
            int(obj.destroyed),  # 2
            rel_pt.r / agent.radar_radius,  # 3
            np.deg2rad(rel_pt.theta),  # 4
            0,  # 5
            0,  # 6
            0,  # 7
            0,  # 8
            0,  # 9
            0,  # 10
        ]
        return np.array(obs)

    def gen_bullseye_obs(self, agent: Aircraft, obj: Bullseye):
        rel_pt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
        obs = [
            OBJECT_TO_IDX[obj.type],  # 0
            0,  # 1
            int(obj.destroyed),  # 2
            rel_pt.r / agent.radar_radius,  # 3
            np.deg2rad(rel_pt.theta),  # 4
            0,  # 5
            0,  # 6
            0,  # 7
            0,  # 8
            0,  # 9
            0,  # 10
        ]
        return np.array(obs)

    @classmethod
    def gen_missile_obs(cls, agent: Aircraft, obj: Missile):
        rel_polar_wpt = agent.waypoint.relative_polar_waypoint(other=obj.waypoint)
        obs = [
            OBJECT_TO_IDX[obj.type],  # 0
            int(obj.color != agent.color),  # 1
            int(obj.destroyed),  # 2
            rel_polar_wpt.r / agent.radar_radius,  # 3
            np.deg2rad(rel_polar_wpt.theta),  # 4
            np.deg2rad(rel_polar_wpt.phi),  # 5
            obj.speed / agent.radar_radius,  # 6
            obj.turn_radius / agent.radar_radius,  # 7
            obj.fuel / agent.options.missile_fuel_capacity,  # 8
            0,  # 9
            0,  # 10
        ]

        if obj.color != agent.color:
            if agent.options.obs_ignore_enemy_missile_fuel:
                # 不知道敌方导弹的剩余油量
                obs[8] = -1
        return obs

    def gen_obs(self, agent_name: str):
        """
        获取agent视角的obs
        注意这里的坐标用相对极坐标来表示
        :return: np.ndarray
        """
        agent = self.battle_area.get_agent(agent_name)
        obs = np.zeros(self.observation_space.shape)

        obs[0, :] = self.gen_self_obs(agent)
        index = 1
        # 隐藏掉雷达探测范围以外的obs，并且移除一些无法获取的信息
        # 飞机
        for obj in self.battle_area.agents:
            if obj.name == agent.name:
                continue
            if agent.options.obs_ignore_destroyed and obj.destroyed:
                continue
            if not agent.options.obs_ignore_radar and agent.distance(obj) > agent.radar_radius:
                # 在自己的雷达范围之外
                if agent.options.obs_allow_memory:
                    # 允许使用记忆
                    memory_key = f'{agent.name}-{obj.name}'
                    if memory_key in self.cache:
                        obj = self.cache[memory_key]
                        obs[index, :] = self.gen_aircraft_obs(agent=agent, obj=obj, is_memory=True)
                        index += 1
                continue
            obs[index, :] = self.gen_aircraft_obs(agent=agent, obj=obj, is_memory=False)
            index += 1
            if agent.options.obs_allow_memory:
                self.cache[f'{agent.name}-{obj.name}'] = obj.__copy__()

        # 基地默认是知道的（不考虑雷达）
        for obj in self.battle_area.homes:
            obs[index, :] = self.gen_home_obs(agent=agent, obj=obj)
            index += 1

        # 牛眼
        obs[index, :] = self.gen_bullseye_obs(agent=agent, obj=self.battle_area.bullseye)
        index += 1

        # 导弹
        for obj in self.battle_area.missiles:
            if obj.destroyed:
                continue
            if index >= len(obs):
                break
            if not agent.options.obs_ignore_radar and agent.distance(obj) > agent.radar_radius:
                # 探测范围之外的导弹
                continue
            obs[index, :] = self.gen_missile_obs(agent=agent, obj=obj)
            index += 1

        return obs

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
