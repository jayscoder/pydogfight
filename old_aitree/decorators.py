from __future__ import annotations
from .behaviour import *
from .colors import *
import time


@register(tag=['Decorator'], type=NODE_TYPE.DECORATOR, color=COLORS.DEFAULT_DECORATOR)
class Decorator(Behaviour):
    """
    装饰节点
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decorated = self.children[0]
        self.decorated.parent = self

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("DecoratorNode must have one child node")

    def tick(self) -> typing.Iterator[Behaviour]:
        """
        Manage the decorated child through the tick.

        Yields:
            a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # initialise just like other behaviours/composites
        if self.status != RUNNING:
            self.initialise()
        # interrupt proceedings and process the child node
        # (including any children it may have as well)
        for node in self.decorated.tick():
            yield node
        # resume normal proceedings for a Behaviour's tick
        new_status = self.execute()
        if new_status not in list(common.Status):
            self.logger.error(
                    "A behaviour returned an invalid status, setting to INVALID [%s][%s]"
                    % (new_status, self.name)
            )
            new_status = common.Status.INVALID
        if new_status != common.Status.RUNNING:
            self.stop(new_status)
        self.status = new_status
        yield self


@register(tag=['Invert', 'Not'])
class Invert(Decorator):
    """
    取反节点
    """

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("Invert node must have one child node")

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            return INVALID

        status = self.children[0].tick()
        if status == SUCCESS:
            return FAILURE
        elif status == FAILURE:
            return SUCCESS
        else:
            return status

    def effect_score(self) -> float:
        return 1 - self.children[0].effect_score()


Not = Invert  # 别名


@register
class ForceSuccess(Decorator):
    """
    无条件返回成功节点，执行孩子节点，无论孩子节点返回什么结果，都向父节点返回 SUCCESS
    """

    def execute(self) -> NODE_STATUS:
        for child in self.children:
            child.tick()
        return SUCCESS

    def effect_score(self) -> float:
        return 1


@register
class ForceFail(Decorator):
    """
    无条件返回失败节点
    """

    def execute(self) -> NODE_STATUS:
        for child in self.children:
            child.tick()
        return FAILURE

    def effect_score(self) -> float:
        return 0

#
# @register
# class ForceRunning(Decorator):
#     """
#     无条件返回正在运行节点
#     """
#
#     def execute(self) -> NODE_STATUS:
#         for child in self.children:
#             child.tick()
#         return RUNNING
#
#     def effect_score(self) -> float:
#         return 0
#
#
# @register
# class ForceNotRun(Decorator):
#     """
#     无条件返回不运行节点，同时跳过孩子节点
#     """
#
#     def execute(self) -> NODE_STATUS:
#         for child in self.children:
#             child.skip()
#         return INVALID
#
#     def effect_score(self) -> float:
#         return 0

#
# @register(
#         props=[{
#             'name'    : 'count',
#             'type'    : int,
#             'default' : 1,
#             'required': True,
#             'desc'    : '重复执行节点，最多执行孩子节点count次'
#         }]
# )
# class Repeat(Decorator):
#     """
#     重复执行节点
#     最多执行孩子节点count次，（count作为数据输入），直到孩子节点返回失败，则该节点返回FAILURE，若孩子节点返回RUNNING ，则同样返回RUNNING。
#     """
#
#     @property
#     def count(self):
#         return self.get_prop('count')
#
#     def validate(self):
#         super().validate()
#         if len(self.children) != 1:
#             raise ValidateError("Repeat node must have one child node")
#
#     def execute(self) -> NODE_STATUS:
#         status = INVALID
#         for i in range(self.count):
#             status = self.children[0].tick()
#             if status == FAILURE or status == RUNNING:
#                 break
#         return status
#
#
# @register
# class Retry(Decorator):
#     """
#     重试节点
#     最多执行孩子节点count次，（count作为数据输入），直到孩子节点返回成功，则该节点返回SUCCESS，若孩子节点返回RUNNING ，则同样返回RUNNING。
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.count = kwargs.get('count', 1)
#
#     def validate(self):
#         super().validate()
#         if len(self.children) != 1:
#             raise ValidateError("Retry node must have one child node")
#
#     def execute(self) -> NODE_STATUS:
#         status = INVALID
#         for i in range(self.count):
#             status = self.children[0].tick()
#             if status == SUCCESS or status == RUNNING:
#                 break
#         return status
#
#
# @register
# class UntilFail(Decorator):
#     """
#     直到失败节点
#     执行孩子节点，如果节点返回结果不是 Fail，向父节点返回 Running，直到节点返回 Fail，向父节点返回 Success
#     """
#
#     def validate(self):
#         super().validate()
#         if len(self.children) != 1:
#             raise ValidateError("UntilFail node must have one child node")
#
#     def execute(self) -> NODE_STATUS:
#         status = self.children[0].tick()
#         if status == FAILURE:
#             return SUCCESS
#         else:
#             return RUNNING
#
#
# @register
# class UntilSuccess(Decorator):
#     """
#     直到成功节点
#     执行孩子节点，如果节点返回结果不是 SUCCESS，向父节点返回 RUNNING，直到节点返回 SUCCESS，向父节点返回 SUCCESS
#     """
#
#     def validate(self):
#         super().validate()
#         if len(self.children) != 1:
#             raise Exception("UntilSuccess node must have one child node")
#
#     def execute(self) -> NODE_STATUS:
#         status = self.children[0].tick()
#         if status == SUCCESS:
#             return SUCCESS
#         else:
#             return RUNNING
#
#
# @register(
#         props={
#             'seconds': {
#                 'type'   : float,
#                 'default': 0,
#                 'desc'   : '节流时间间隔，单位秒'
#             },
#             'ticks'  : {
#                 'type'   : int,
#                 'default': 0,
#                 'desc'   : '节流tick间隔'
#             }
#         }
# )
# class Throttle(Decorator):
#     """
#     节流节点
#     在指定时间内，只执行一次孩子节点（其他时候直接返回上次执行结果），如果孩子节点返回 RUNNING，下次执行时，直接返回 RUNNING
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.last_execute_time = -1e6
#         self.last_execute_tick = -1e6
#         self.last_status = INVALID
#
#     @property
#     def seconds(self):
#         return self.get_prop('seconds')
#
#     def validate(self):
#         super().validate()
#         if len(self.children) != 1:
#             raise ValidateError("Throttle node must have one child node")
#
#     def execute(self) -> NODE_STATUS:
#         seconds = self.get_prop('seconds')
#         ticks = self.get_prop('ticks')
#
#         if seconds > 0:
#             now = time.time()
#             if now - self.last_execute_time < seconds:
#                 return self.last_status
#             self.last_execute_time = now
#         elif ticks > 0:
#             if self.tick_count - self.last_execute_tick < ticks:
#                 return self.last_status
#             self.last_execute_tick = self.tick_count
#
#         self.last_status = self.children[0].tick()
#         return self.last_status
