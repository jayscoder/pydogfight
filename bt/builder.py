import pydogfight

from pydogfight.policy.bt.builder import BTPolicyBuilder
from bt.nodes import *
from bt.nodes_rl_1v1 import *
from bt.nodes_rl_multiple import *
from bt.nodes_v7 import *
from bt.nodes_v8 import *


class CustomBTBuilder(BTPolicyBuilder):

    def register_default(self):
        super().register_default()
        # 在这里注册新的行为节点，不然没办法在xml中使用
        self.register_node(
                TestNode,
                RandomInitWaypointNearGameCenter,
                CheatGoToNearestEnemyWithMemory,
                RLGoToLocation1V1,
                RLCondition1V1,
                RLFireAndGoToLocation1V1,

                RLGoToLocationMultiple,
                RLFireAndGoToLocationMultiple,
                RLConditionMultiple,
        )

        # V7
        self.register_node(
                V7Init,
                V7SACGoToLocation1V1,
                V7SACFireAndGoToLocation1V1,
                V7SACCondition1V1
        )

        # V8
        self.register_node(
                V8Init,
                V8SACFloatArrayValue,
                V8SACFloatValue,
                V8SACSwitcher,
                V8SACFireAndGoToLocation1V1,
                V8PPOSwitcher,
                V8PPOFireAndGoToLocation1V1,
                V8PPOGoToLocation1V1,
                V8SACGoToLocation1V1,
                V8PPOFloatArray
        )
