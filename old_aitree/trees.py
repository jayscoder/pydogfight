from __future__ import annotations
from .composites import Parallel
from .register import *
from .behaviour import *
from .colors import *


@register(type=NODE_TYPE.BEHAVIOR_TREE, order=0, color=COLORS.DEFAULT_BEHAVIOR_TREE)
class BehaviorTree(Parallel):

    @property
    def label(self):
        return self.get_prop('label') or self.id

    @label.setter
    def label(self, value):
        self.attributes['label'] = value
