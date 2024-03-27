from __future__ import annotations

from .config import *
from .behaviour import Behaviour


class NotFound(Behaviour):
    """
    未找到节点，用来作为占位节点使用
    """

    def validate(self):
        super().validate()
        raise ValidateError('node not found')

    def execute(self) -> NODE_STATUS:
        print(f'node {self.label} not found: id={self.id}')
        return INVALID

# @register
# class RandomWeighted(CompositeNode):
#     """
#     随机权重执行节点
#     """
#
#     def execute(self) -> NODE_STATUS:
#         pass


# @behavior(key='namespace')
# class NamespaceNode(Parallel):
#     """
#     命名空间节点，执行方式类似于并行节点
#     """
#
#     def __init__(self, name: str, **kwargs):
#         super().__init__(**kwargs)
#         self.name = name
#
#     def validate(self) -> Tuple[bool, str]:
#         """
#         组合节点的所有子节点都必须是行为节点/条件节点/组合节点/装饰节点/变量节点 中的其中一个
#         :return:
#         """
#         for child in self.children:
#             if child.type not in [NODE_TYPE.ACTION, NODE_TYPE.CONDITION, NODE_TYPE.COMPOSITE, NODE_TYPE.DECORATOR,
#                                   NODE_TYPE.VARIABLE]:
#                 return False, "NamespaceNode's child must be ActionNode/ConditionNode/CompositeNode/DecoratorNode/VariableNode"
#             ok, msg = child.validate()
#             if not ok:
#                 return False, msg
#         return True, ''
#
#     def execute(self) -> NODE_STATUS:
#         if len(self.children) == 0:
#             # 没有子节点，直接返回成功
#             return SUCCESS
#
#         success_threshold = self.success_threshold
#         if success_threshold == -1:
#             # 所有子节点都成功才算成功
#             success_threshold = len(self.children)
#
#         success_count = 0
#         fail_count = 0
#         running_indexes = []
#
#         for index, child in enumerate(self.children):
#             if len(self.running_indexes) > 0 and index not in self.running_indexes:
#                 # 如果当前存在正在运行列表 且 当前节点不在正在运行的节点列表中，则跳过
#                 child.skip()
#                 continue
#
#             child_status = child.do_execute()
#             if child_status == RUNNING:
#                 running_indexes.append(index)
#             elif child_status == SUCCESS:
#                 success_count += 1
#             elif child_status == FAILURE:
#                 fail_count += 1
#
#         self.running_indexes = running_indexes
#         if success_count >= success_threshold:
#             return SUCCESS
#         elif len(running_indexes) > 0:
#             return RUNNING
#         else:
#             return FAILURE
