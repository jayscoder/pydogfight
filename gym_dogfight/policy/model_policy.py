import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from .policy import AgentPolicy
from gym_dogfight.envs import Dogfight2dEnv


class ModelPolicy(AgentPolicy):

    def __init__(self, model: BaseAlgorithm, env: Dogfight2dEnv, agent_name: str, delta_time: float = 1):
        super().__init__(env=env, agent_name=agent_name, delta_time=delta_time)
        self.model = model
        self.state = None

    def execute(self, observation, delta_time: float):
        action, self.state = self.model.predict(observation, deterministic=True, state=self.state)
        self.actions.put_nowait(action)
