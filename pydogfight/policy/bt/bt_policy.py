from __future__ import annotations

import pybts

from pydogfight.policy.policy import Policy, AgentPolicy
from pydogfight.envs import Dogfight2dEnv


class BTPolicy(AgentPolicy):
    def __init__(self,
                 tree: pybts.Tree,
                 env: Dogfight2dEnv,
                 agent_name: str,
                 update_interval: float = 1,
                 ):
        super().__init__(env=env, agent_name=agent_name, update_interval=update_interval)
        self.tree = tree
        self.share_cache = { }

    def _setup(self):
        super()._setup()
        self.tree.setup(env=self.env, agent_name=self.agent_name, share_cache=self.share_cache, actions=self.actions)

    def reset(self):
        super().reset()
        self.share_cache.clear()
        self.tree.reset()

    def execute(self, observation, delta_time: float):
        self.tree.tick()
