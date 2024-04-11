import pydogfight

from pydogfight.policy.bt.builder import BTPolicyBuilder
from bt.nodes import *


class CustomBTBuilder(BTPolicyBuilder):

    def register_default(self):
        super().register_default()
        # 在这里注册新的行为节点，不然没办法在xml中使用
        self.register_node(
                TestNode
        )
