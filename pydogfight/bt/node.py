from __future__ import annotations

from py_trees.behaviour import Behaviour
from abc import ABC


class BTNode(Behaviour, ABC):
    def __init__(self, name: str):
        super().__init__(name=name)

    def reset(self):
        pass

    @classmethod
    def creator(cls, d, c):
        return cls(name=d['name'])


