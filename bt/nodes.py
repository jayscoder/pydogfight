from __future__ import annotations

import typing

from pydogfight.policy.bt import BTPolicyNode
from pydogfight import Dogfight2dEnv, Aircraft, Options

class TestNode(BTPolicyNode):
    """定义的新的节点，需要在builder中注册才能在xml中使用"""

    def __init__(self, count: int | str = '', **kwargs):
        super(TestNode, self).__init__(**kwargs)  # 必须得给父节点传递kwargs参数，不然会报错
        self.count = count  # 自定义的参数在使用xml构建的时候传进来的一定是str，后面需要在setup的时候通过converter进行数值转换

    def setup(self, **kwargs: typing.Any) -> None:
        super(TestNode, self).setup(**kwargs)
        self.count = self.converter.int(self.count)
        # jinjia2语法需要通过self.converter.render(...)来进行转换

    def to_data(self):
        # 这里的数据会在track的时候保存下来
        return {
            **super().to_data(),
            'agent': self.agent.to_dict()
        }
