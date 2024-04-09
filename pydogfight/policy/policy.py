from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
from pydogfight.envs import Dogfight2dEnv
from queue import Queue
from pydogfight.core.world_obj import Aircraft


class Policy(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplemented

    @abstractmethod
    def take_action(self):
        raise NotImplemented

    @abstractmethod
    def put_action(self):
        raise NotImplementedError


class AgentPolicy(Policy, ABC):

    def __init__(self, env: Dogfight2dEnv, agent_name: str, update_interval: float = 0):
        super().__init__()
        self.env = env
        self.options = env.options
        self.update_interval = update_interval
        self.last_time = 0
        self.actions = Queue()
        self._has_setup = False
        self.agent_name = agent_name

    def _setup(self):
        self._has_setup = True
        pass

    def reset(self):
        if not self._has_setup:
            self._has_setup = True
            self._setup()
        self.last_time = 0
        while not self.actions.empty():
            # 清空actions
            self.actions.get_nowait()

    @property
    def agent_index(self):
        for i, name in enumerate(self.options.agents()):
            if name == self.agent_name:
                return i

    @property
    def agent(self) -> Aircraft:
        return self.env.get_agent(self.agent_name)

    def put_action(self):
        while not self.actions.empty():
            action = self.actions.get_nowait()
            self.agent.put_action(action)

    def take_action(self):
        # 根据当前状态选择动作，obs的第一个是自己，确保策略更新满足时间间隔
        delta_time = self.env.time - self.last_time
        if self.update_interval == 0 and delta_time == 0:
            return
        if self.update_interval > 0 and delta_time < self.update_interval:
            return
        self.last_time = self.env.time
        self.execute(observation=self.env.gen_agent_obs(agent_name=self.agent_name), delta_time=delta_time)

    @abstractmethod
    def execute(self, observation, delta_time: float):
        """
        执行具体策略
        Args:
            observation:
            delta_time:

        Returns:

        """
        pass
    

class MultiAgentPolicy(Policy):
    def __init__(self, policies: list[Policy]):
        self.policies = policies

    def reset(self):
        for policy in self.policies:
            policy.reset()

    def put_action(self):
        for policy in self.policies:
            policy.put_action()

    def take_action(self):
        for policy in self.policies:
            policy.take_action()
