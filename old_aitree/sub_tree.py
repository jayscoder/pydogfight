from __future__ import annotations
from .behaviour import *
from .colors import *
import time
from typing import Tuple


@register(
        label='嵌套节点',
        props={
            'ref_id': {
                'type'    : 'string',
                'default' : '',
                'required': True,
                'desc'    : '嵌套节点引用的BehaviorTree的ID'
            }
        },
        type=NODE_TYPE.SUB_TREE,
        color=COLORS.DEFAULT_SUB_TREE
)
class SubTree(Behaviour):
    """
    子树
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ref_node = None

    @property
    def ref_id(self):
        return self.get_prop('ref_id', once=True)

    @property
    def ref_node(self):
        from .register import find_global_node
        if self._ref_node is not None:
            return self._ref_node

        ref_id = self.ref_id
        self._ref_node = self.root.find_node(ref_id) or find_global_node(ref_id)
        if self._ref_node is None:
            raise Exception(f"SubTree id {ref_id} not found")
        self._ref_node = self._ref_node(**self.attributes)
        self._ref_node.id = self.id
        self._ref_node.context = self.context

        # 修改所有的子ID, 以防止重复, 所有的子ID都加上父ID作为前缀
        def traverse(node):
            node.id = f'{self.id}|{node.id}'
            for _child in node.children:
                traverse(_child)

        for child in self._ref_node.children:
            traverse(child)

        return self._ref_node

    def validate(self) -> Tuple[bool, str]:
        from .register import find_global_node
        super().validate()
        if len(self.children) != 0:
            raise ValidateError("SubTree can not have child node")
        ref_id = self.ref_id
        if ref_id == '':
            raise ValidateError(f"SubTree {ref_id}: id can not be empty")
        if ref_id == self.id:
            raise ValidateError(f"SubTree {ref_id}: id can not be self")
        # 引用的节点不能是自己的父亲节点
        par = self.parent
        while par is not None:
            if par.id == ref_id:
                raise ValidateError(f"SubTree {ref_id}: can not reference self parent")
            par = par.parent
        # 校验引用的节点是否存在
        ref_node = self.root.find_node(ref_id) or find_global_node(ref_id)
        if ref_node is None:
            raise ValidateError(f"SubTree {ref_id}: id not found")
        return True, ''

    def execute(self) -> NODE_STATUS:
        if self.ref_node is None:
            return FAILURE
        return self.ref_node.tick()

    def effect_score(self) -> float:
        return self.ref_node.effect_score()
