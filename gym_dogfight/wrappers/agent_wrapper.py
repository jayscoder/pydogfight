from __future__ import annotations
from gymnasium.core import ObsType, WrapperObsType

from gym_dogfight.envs import Dogfight2dEnv
from gym_dogfight.core.options import Options
from gym_dogfight.core.constants import *
import gymnasium as gym
import numpy as np
from gym_dogfight.policy import Policy
from gym_dogfight.core.actions import *


class AgentWrapper(gym.Wrapper):
    """
    指定单个Agent自己的视角
    """

    def __init__(self, env: Dogfight2dEnv, agent_name: str = ''):
        super().__init__(env)
        self.env = env
        self.options = env.options
        if agent_name == '':
            agent_name = self.options.red_agents[0]
        self.agent_name = agent_name
        for i, agent in enumerate(self.options.agents):
            if agent == agent_name:
                self.agent_index = i
                break
        self.action_space = gym.spaces.Box(
                low=np.array([0, -int(self.options.game_size[0] / 2), -int(self.options.game_size[1] / 2)]),
                high=np.array([len(Actions), int(self.options.game_size[0] / 2), int(self.options.game_size[1] / 2)]),
                shape=(3,),
                dtype=np.int32)
        self.observation_space = env.observation_space

    def step(self, action):
        agent = self.env.get_agent(self.agent_name)
        agent.put_action(action)
        info = self.env.gen_info()
        obs = self.env.gen_agent_obs(agent_name=self.agent_name)
        reward = self.env.gen_reward(color=agent.color)
        return obs, reward, info['terminated'], info['truncated'], info

    def action(self, action):
        actions = np.zeros(self.env.action_space.shape)
        actions[self.agent_index] = action
        return actions

# class PolicyWrapper(gym.Wrapper):
#     def __init__(self, env: Dogfight2dEnv, policies: list[Policy]):
#         super().__init__(env)
#         self.env = env
#         self.policies = policies
#         self.options = env.options
#
#     def step(self, action):
#         for p in self.policies:
#             p.select_action(observation=self.env.gen_obs())
#             p.put_action()
#         obs, reward, terminated, truncated, info = self.env.step(self.env.empty_action())
#         return obs, reward, terminated, truncated, info
