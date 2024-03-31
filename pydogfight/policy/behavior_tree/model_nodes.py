from __future__ import annotations
from pydogfight.policy.behavior_tree.nodes import *
from stable_baselines3 import PPO
from pybts import Status
from pydogfight.core.actions import Actions


class BTPPOModel(BTPolicyAction):
    def __init__(self, model: str | PPO, name: str=''):
        super().__init__(name=name)
        if isinstance(model, str):
            self.model_path = model
            self.model = PPO.load(path=model)
        else:
            self.model_path = ''
            self.model = model

    @classmethod
    def creator(cls, d: dict, c: list):
        return cls(
                model=d['model'],
                name=d['name']
        )

    def to_data(self):
        return {
            **super().to_data(),
            'model_path': self.model_path,
            'model': self.model
        }

    def update(self) -> Status:
        obs = self.env.gen_agent_obs(agent_name=self.agent_name)
        action, _ = self.model.predict(obs, deterministic=True, state=None)
        relative_waypoint = self.agent.waypoint.relative_waypoint(dx=action[1] * self.agent.radar_radius,
                                                                  dy=action[2] * self.agent.radar_radius)
        action_type = Actions.extract_action_in_value_range(actions=None, value=action[0], value_range=(-1, 1))
        self.actions.put_nowait((action_type, relative_waypoint.x, relative_waypoint.y))
        return Status.SUCCESS


class BTPPOGoToLocationModel(BTPPOModel):

    def update(self) -> Status:
        obs = self.env.gen_agent_obs(agent_name=self.agent_name)
        action, _ = self.model.predict(obs, deterministic=True, state=None)
        relative_waypoint = self.agent.waypoint.relative_waypoint(dx=action[0] * self.agent.radar_radius,
                                                                  dy=action[1] * self.agent.radar_radius)

        self.actions.put_nowait((Actions.go_to_location, relative_waypoint.x, relative_waypoint.y))
        return Status.SUCCESS


class BTPPOFireMissileModel(BTPPOModel):

    def update(self) -> Status:
        self.model.action_space
        obs = self.env.gen_agent_obs(agent_name=self.agent_name)
        action, _ = self.model.predict(obs, deterministic=True, state=None)
        relative_waypoint = self.agent.waypoint.relative_waypoint(dx=action[0] * self.agent.radar_radius,
                                                                  dy=action[1] * self.agent.radar_radius)

        self.actions.put_nowait((Actions.fire_missile, relative_waypoint.x, relative_waypoint.y))
        self.model.train()
        return Status.SUCCESS
