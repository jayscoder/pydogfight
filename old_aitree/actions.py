from __future__ import annotations
from .behaviour import *
import time


@register(type=NODE_TYPE.ACTION, visible=False)
class Action(Behaviour):
    """
    行为节点
    """

    def validate(self):
        super().validate()
        if len(self.children) != 0:
            raise ValidateError("ActionNode can not have child node")


@register(desc='设置黑板变量', props=[
    {
        'name'    : 'key',
        'type'    : 'string',
        'default' : '',
        'required': True,
        'desc'    : '变量名'
    },
    {
        'name'    : 'type',
        'type'    : 'string',
        'default' : 'string',
        'required': False,
        'desc'    : '变量类型'
    },
    {
        'name'    : 'value',
        'type'    : 'string',
        'default' : '',
        'required': False,
        'desc'    : '变量值'
    },
    {
        'name'    : 'once',
        'type'    : 'bool',
        'default' : False,
        'required': False,
        'desc'    : '是否只设置一次'
    }
])
class SetBlackboard(Action):
    """
    设置黑板变量节点
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_set = False

    def validate(self):
        if self.key == '':
            raise ValidateError("key can not be empty")

    def execute(self) -> NODE_STATUS:
        if self.once and self.is_set:
            return SUCCESS
        self.blackboard[self.key] = self.value
        self.is_set = True
        return SUCCESS

    def effect_score(self) -> float:
        return 1

    @property
    def once(self):
        return self.get_prop('once')

    @property
    def key(self):
        return self.get_prop('key')

    @property
    def type(self):
        return self.get_prop('type')

    @property
    def value(self):
        value = self.get_prop('value')
        return common.parse_type_value(value, value_type=self.type)


@register(desc='打印', props=[{
    "name"   : "msg",
    "type"   : str,
    "desc"   : "要打印的内容",
    "default": ""
}])
class Print(Action):

    @property
    def msg(self):
        return self.get_prop(['msg', 'message'])

    def execute(self) -> NODE_STATUS:
        print(self.msg)
        return SUCCESS

    def effect_score(self) -> float:
        return 1


@register(label='随机动作', desc='随机执行一个仿真动作')
class RandomAction(Action):
    """
    随机执行一个动作
    """

    def execute(self) -> NODE_STATUS:
        self.simulation.step(action=None)
        return SUCCESS

    def effect_score(self) -> float:
        return 1


@register
class DoAction(Action):
    """
    执行一个动作
    """

    @property
    def action(self):
        return self.get_prop('action')

    def execute(self) -> NODE_STATUS:
        self.context.step(action=self.action)
        return SUCCESS

    def effect_score(self) -> float:
        return 1


@register(label='睡眠', desc='睡眠一段时间', props={
    'seconds': {
        'type'   : float,
        'default': 1,
        'desc'   : '睡眠时间，单位秒'
    }
})
class Sleep(Action):
    """
    睡眠节点
    """

    @property
    def seconds(self):
        return self.get_prop('seconds')

    def execute(self) -> NODE_STATUS:
        time.sleep(self.seconds)
        return SUCCESS

    def effect_score(self) -> float:
        return 1
