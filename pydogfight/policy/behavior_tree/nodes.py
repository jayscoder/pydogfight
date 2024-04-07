from __future__ import annotations

import queue

import pybts

from pydogfight.policy.policy import Policy, AgentPolicy
from pydogfight.envs import Dogfight2dEnv, Aircraft
from abc import ABC


class BTPolicyNode(pybts.Node, ABC):
    """
    BT Policy Base Class Node
    """

    def __init__(self, name: str = ''):
        super().__init__(name=name)
        self.env: Dogfight2dEnv | None = None
        self.share_cache = { }
        self.agent_name = ''
        self.update_messages = queue.Queue(maxsize=20)  # update过程中的message

    @property
    def agent(self) -> Aircraft | None:
        if self.env is None:
            return None
        return self.env.get_agent(self.agent_name)

    def put_update_message(self, msg: str):
        if self.update_messages.full():
            self.update_messages.get_nowait()
        msg = f"{self.debug_info['tick_count']}: {msg}"
        self.update_messages.put_nowait(msg)

    def to_data(self):
        return {
            **super().to_data(),
            'agent_name'     : self.agent_name,
            'update_messages': pybts.utility.read_queue_without_destroying(self.update_messages)
        }


class BTPolicyAction(BTPolicyNode, pybts.Action, ABC):
    """BT Policy Action Node"""

    def to_data(self):
        return {
            **super().to_data(),
            **pybts.Action.to_data(self)
        }


class BTPolicyCondition(BTPolicyNode, pybts.Condition, ABC):
    """
    BT Policy Condition Node
    """

    def to_data(self):
        return {
            **super().to_data(),
            **pybts.Condition.to_data(self)
        }


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
        for node in self.tree.root.iterate():
            if isinstance(node, BTPolicyNode):
                node.share_cache = self.share_cache
                node.env = env
                node.agent_name = agent_name
            if isinstance(node, pybts.Action):
                node.actions = self.actions

    def _setup(self):
        super()._setup()
        self.tree.setup()

    def reset(self):
        super().reset()
        self.share_cache.clear()
        self.tree.reset()

    def execute(self, observation, delta_time: float):
        self.tree.tick()
