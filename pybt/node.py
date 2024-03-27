from __future__ import annotations

from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Parallel, Selector
from py_trees.decorators import Inverter
from py_trees.trees import BehaviourTree
from py_trees.visitors import VisitorBase
from abc import ABC


class BTNode(Behaviour, ABC):
    def __init__(self, name: str):
        super().__init__(name=name)

    def reset(self):
        pass

    @classmethod
    def creator(cls, d: dict, c: list):
        return cls(name=d['name'])

    def view_info(self):
        # 在viewer上查看的信息
        return { }
