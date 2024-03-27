from __future__ import annotations

import py_trees
from abc import ABC
from typing import Callable
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import copy
from pydogfight.policy.policy import Policy, AgentPolicy
from py_trees.behaviour import Behaviour
from py_trees.composites import *
from py_trees.common import Status
from py_trees.trees import BehaviourTree
from py_trees import visitors
from queue import Queue
from pydogfight.envs import Dogfight2dEnv, Aircraft
from abc import ABC
from pydogfight.bt.node import BTNode


class BTPolicyNode(BTNode, ABC):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.env: Dogfight2dEnv | None = None
        self.share_cache = { }
        self.agent_name = ''
        self.update_interval = 0
        self.actions: Queue | None = None

    @property
    def agent(self) -> Aircraft:
        return self.env.get_agent(self.agent_name)

    def reset(self):
        pass

    @classmethod
    def creator(cls, d, c):
        return cls(name=d['name'])

    def update(self) -> Status:
        return Status.INVALID


class BTPolicyAction(BTPolicyNode, ABC):
    pass


class BTPolicyCondition(BTPolicyNode, ABC):
    pass


class BTPolicy(AgentPolicy):
    def __init__(self,
                 root: Behaviour,
                 env: Dogfight2dEnv,
                 agent_name: str,
                 update_interval: float = 1,
                 visitor: typing.Optional[visitors.VisitorBase] = None,
                 ):
        super().__init__(env=env, agent_name=agent_name, update_interval=update_interval)
        self.tree = BehaviourTree(root)
        self.share_cache = { }
        for node in self.tree.root.iterate():
            if isinstance(node, BTPolicyNode):
                node.share_cache = self.share_cache
                node.env = env
                node.agent_name = agent_name
                node.update_interval = update_interval
                node.actions = self.actions
        if visitor is not None:
            self.tree.add_visitor(visitor)

    def _setup(self):
        self.tree.setup()

    def reset(self):
        super().reset()
        for node in self.tree.root.iterate():
            if isinstance(node, BTNode):
                node.reset()

    def execute(self, observation, delta_time: float):
        self.tree.tick()
