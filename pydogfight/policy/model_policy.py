import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from .policy import AgentPolicy
from pydogfight.envs import Dogfight2dEnv


class ModelPolicy(AgentPolicy):

    def __init__(self, model: BaseAlgorithm, env: Dogfight2dEnv, agent_name: str, update_interval: float = 1):
        super().__init__(env=env, agent_name=agent_name, update_interval=update_interval)
        self.model = model
        self.state = None

    def execute(self, observation, delta_time: float):
        action, self.state = self.model.predict(observation, deterministic=True, state=self.state)
        self.actions.put_nowait(action)
