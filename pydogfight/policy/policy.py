from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Optional
from pydogfight.envs import Dogfight2dEnv
import numpy as np
from queue import Queue
from collections import defaultdict
import random
from pydogfight.core.actions import *
from pydogfight.core.world_obj import Aircraft


class Policy(ABC):
    def __init__(self, env: Dogfight2dEnv, update_interval: float = 1):
        self.env = env
        self.options = env.options
        self.update_interval = update_interval
        self.last_time = 0
        self.actions = Queue()
        self._has_setup = False

    def reset(self):
        self.last_time = 0
        while not self.actions.empty():
            # 清空actions
            self.actions.get_nowait()
        if not self._has_setup:
            self._has_setup = True
            self._setup()

    def _setup(self):
        pass

    def select_action(self):
        # 根据当前状态选择动作，obs的第一个是自己
        delta_time = self.env.time - self.last_time
        if delta_time < self.update_interval:
            return
        self.last_time = self.env.time
        self.execute(observation=self.gen_obs(), delta_time=delta_time)

    @abstractmethod
    def gen_obs(self):
        raise NotImplementedError

    @abstractmethod
    def execute(self, observation, delta_time: float):
        pass

    @abstractmethod
    def put_action(self):
        raise NotImplementedError


class AgentPolicy(Policy, ABC):

    def __init__(self, env: Dogfight2dEnv, agent_name: str, update_interval: float = 1):
        super().__init__(env=env, update_interval=update_interval)
        self.agent_name = agent_name

    @property
    def agent_index(self):
        for i, name in enumerate(self.options.agents):
            if name == self.agent_name:
                return i

    @property
    def agent(self) -> Aircraft:
        return self.env.get_agent(self.agent_name)

    def put_action(self):
        while not self.actions.empty():
            action = self.actions.get_nowait()
            self.agent.put_action(action)

    def gen_obs(self):
        return self.env.gen_agent_obs(agent_name=self.agent_name)


class MultiAgentPolicy(Policy):
    def __init__(self, env: Dogfight2dEnv, policies: List[AgentPolicy]):
        super().__init__(env=env, update_interval=0)
        self.policies = policies

    def reset(self):
        super().reset()
        for policy in self.policies:
            policy.reset()

    def put_action(self):
        while not self.actions.empty():
            self.env.put_action(action=self.actions.get_nowait())

    def execute(self, observation, delta_time: float):
        for p in self.policies:
            p.select_action()
        over = [p.actions.empty() for p in self.policies]

        while not all(over):
            act_value = self.env.empty_action()
            for i, p in enumerate(self.policies):
                if over[i]:
                    continue
                if p.actions.empty():
                    over[i] = True
                    continue
                act_value[p.agent_index, :] = p.actions.get_nowait()
            self.actions.put_nowait(act_value)

    def gen_obs(self):
        return self.env.gen_obs()
