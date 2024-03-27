# 节点状态 Enum
# - 运行中
# - 失败
# - 成功
from typing import List, Dict, Any, Union, Optional
import os

MAX_TREE_DEPTH = os.environ.get('AITREE_MAX_TREE_DEPTH', 1000)  # 最大树深度

# 元数据模版
META_TEMPLATE = {
    'tag'     : [],  # 节点标签（用在xml中），可以是字符串或者字符串列表，如果是字符串列表，则表示该节点可以有多个标签
    'desc'    : '',  # 节点描述
    'props'   : [],  # 节点参数约束
    'label'   : '',  # 节点显示在webui上的标签
    'type'    : '',  # 节点类型
    'visible' : True,  # 节点是否在面板上可见
    'disabled': False,  # 节点是否禁用
    'filepath': '',  # 节点所在的文件路径
    'color'   : '',  # 节点颜色
    'order'   : 10000,  # 节点排序，用于排序节点在面板上的显示顺序，越小越靠前
}

PROP_TEMPLATE = {
    'name'    : '',  # 参数名称
    'type'    : '',  # 参数类型
    'default' : '',  # 默认值
    'required': False,  # 是否必填
    'desc'    : '',  # 参数描述
    'options' : [],  # 选项 用于下拉框 仅在type为str时有效 {'name': '选项1', 'value': '1'}
    'visible' : True,  # 是否可见
    'disabled': False,  # 是否禁用
}

OPTIONS_TEMPLATE = {
    'name' : '',  # 选项名称
    'value': '',  # 选项值
}


# BUILD_IN_PROPS = [
#     'id',
#     'name',
#     'type',
#     'desc',
#     'status',
#     'score',
#     'msg',
#     'children',
#     'context',
#     'parent',
#     'root'
# ]

class NODE_STATUS:

    def __init__(self, value: str = '', score: Union[float, None] = None, msg: str = ''):
        self.value = value
        self.msg = msg
        if score is None:
            if self.value == 'success':
                score = 1
            elif self.value == 'failure':
                score = 0
            elif self.value == 'running':
                score = 0.5
            else:
                score = 0
        self.score = score

    @property
    def is_success(self):
        return self.value == 'success'

    @property
    def is_failure(self):
        return self.value == 'failure'

    @property
    def is_running(self):
        return self.value == 'running'

    def __eq__(self, other):
        if isinstance(other, NODE_STATUS):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        elif isinstance(other, float):
            return self.score == other
        else:
            return False

    def __str__(self):
        if self.msg != '':
            return f'{self.value}[{self.score}]: {self.msg}'
        else:
            return f'{self.value}[{self.score}]'

    def __call__(self, *args, **kwargs):
        """
        设置msg
        :param args:
        :param kwargs:
        :return:
        """
        msg = self.msg
        score = self.score

        if len(args) > 0:
            if isinstance(args[0], str):
                msg = args[0]
            elif isinstance(args[0], float):
                score = args[0]
        if 'msg' in kwargs:
            msg = kwargs['msg']
        if 'score' in kwargs:
            score = kwargs['score']
        return NODE_STATUS(value=self.value, score=score, msg=msg)


SUCCESS = NODE_STATUS('success')
FAILURE = NODE_STATUS('failure')
RUNNING = NODE_STATUS('running')
INVALID = NODE_STATUS('invalid')


class NODE_TYPE:
    """
    节点类型
    """
    VIRTUAL = 'virtual'  # 虚拟节点 默认
    SUB_TREE = 'sub_tree'  # 子树
    COMPOSITE = 'composite'  # 组合节点
    ACTION = 'action'  # 行为节点
    CONDITION = 'condition'  # 条件节点
    DECORATOR = 'decorator'  # 修饰节点
    PROCESSOR = 'processor'  # 处理器节点
    SIMULATION = 'simulation'  # 模拟节点
    BEHAVIOR_TREE = 'behavior_tree'  # 行为树节点


# 定义校验Error
class ValidateError(Exception):
    def __init__(self, msg: str = ''):
        self.msg = msg

    def __str__(self):
        return self.msg
