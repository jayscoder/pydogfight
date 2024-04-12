from __future__ import annotations

import queue
import typing

import pybts

from pydogfight.policy.policy import Policy, AgentPolicy
from pydogfight.envs import Dogfight2dEnv, Aircraft
from abc import ABC


class BTPolicyNode(pybts.Action, ABC):
    """
    BT Policy Base Class Node
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_messages = queue.Queue(maxsize=20)  # update过程中的message

    @property
    def env(self) -> Dogfight2dEnv:
        return self.context['env']

    @property
    def agent_name(self) -> str:
        return self.context['agent_name']

    @property
    def agent(self) -> Aircraft | None:
        if self.env is None:
            return None
        return self.env.get_agent(self.agent_name)

    def put_update_message(self, msg: str):
        if not self.env.options.debug:
            return
        if self.update_messages.full():
            self.update_messages.get_nowait()
        msg = f"{self.debug_info['tick_count']}: {msg}"
        self.update_messages.put_nowait(msg)

    def to_data(self):
        from pybts import utility
        return {
            **super().to_data(),
            'agent_name'     : self.agent_name,
            'update_messages': utility.read_queue_without_destroying(self.update_messages)
        }


class BTPolicy(AgentPolicy):
    def __init__(self,
                 tree: pybts.Tree,
                 env: Dogfight2dEnv,
                 agent_name: str,
                 ):
        super().__init__(env=env, agent_name=agent_name)
        self.tree = tree
        env.add_after_reset_handler(lambda _: self.reset())

    def reset(self):
        super().reset()
        self.tree.reset()

    def execute(self, observation, delta_time: float):
        # 更新时间
        self.tree.tick()
        # 收集所有节点的行为，并放到自己的行为库里
        for node in self.tree.root.iterate():
            if isinstance(node, pybts.Action):
                while not node.actions.empty():
                    action = node.actions.get_nowait()
                    self.actions.put_nowait(action)
