from __future__ import annotations
from .behaviour import *
from .colors import *


@register(type=NODE_TYPE.CONDITION, visible=False, color=COLORS.DEFAULT_CONDITION)
class Condition(Behaviour):
    """
    条件节点
    只能向父节点返回 Success或Failed，不得返回运行。
    """

    def validate(self):
        super().validate()
        if len(self.children) != 0:
            raise ValidateError("ConditionNode can not have child node")

    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        :return:
        """
        return self.execute().score
