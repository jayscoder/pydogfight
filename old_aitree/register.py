from __future__ import annotations
from typing import Union, TYPE_CHECKING
from . import utilities
import copy
from .config import *

if TYPE_CHECKING:
    from .behaviour import Behaviour

_TAG_TO_NODES_CLS = { }  # 用于快速查找节点类，key是register时的tag
_CLASSNAME_TO_NODES_CLS = { }  # 用于快速查找节点类，key是node的类名
_ID_TO_NODES = { }  # 用于快速查找节点，key是node的id
REGISTER_NODES = set()  # 注册的节点


def register_node(
        node: Union[type, Behaviour],
        **kwargs
):
    """
    注册节点类
    :param node: 节点类或者节点实例
    :return:
    """
    assert callable(node)

    tag = kwargs.get('tag', '')
    props = kwargs.get('props', [])
    if isinstance(tag, str):
        if tag == '':
            if isinstance(node, type):
                tag = [node.__name__]
            elif isinstance(node, Behaviour):
                tag = [node.tag(deep=True)]
            else:
                tag = [str(node)]
        else:
            tag = [tag]

    if node.meta is None:
        node.meta = copy.deepcopy(META_TEMPLATE)

    props = utilities.parse_props(props)

    if 'props' in node.meta:
        props = utilities.merge_props(props, node.meta['props'])

    node.meta = {
        **node.meta,
        **kwargs,
        'tag'     : tag,
        'props'   : props,
        'visible' : kwargs.get('visible', True),  # 是否在webui上可见，不继承
        'disabled': kwargs.get('disabled', False),  # 是否可用，不继承
        'order'   : kwargs.get('order', 10000),  # 排序，不继承，0表示最前面，-1表示最后面
    }

    all_tags = []
    for _tag in tag:
        _TAG_TO_NODES_CLS[_tag] = node
        all_tags.append(_tag)

    if isinstance(node, type):
        class_name = node.__name__
        snake_class_name = utilities.camel_case_to_snake_case(class_name)
        module_name = f'{node.__module__}.{node.__qualname__}'

        _CLASSNAME_TO_NODES_CLS[class_name] = node
        _CLASSNAME_TO_NODES_CLS[snake_class_name] = node
        _CLASSNAME_TO_NODES_CLS[module_name] = node

        for i_tag in [class_name, snake_class_name, module_name]:
            if i_tag not in all_tags:
                all_tags.append(i_tag)

    node.meta['tag'] = all_tags

    REGISTER_NODES.add(node)
    return node


def register(*args, **kwargs):
    """
    注册节点类
    :param args:
        如果只有一个参数，且为函数，则视为装饰器模式，直接注册
        如果有多个参数，则视为普通模式，需要传入tag, desc, props参数
    :param kwargs:
    :return:
    """

    if len(args) == 1 and callable(args[0]):
        return register_node(node=args[0], **kwargs)

    meta = { }
    for arg in args:
        if isinstance(arg, dict):
            meta = { **meta, **arg }

    meta = { **meta, **kwargs }

    node = kwargs.get('node', None)
    if node is None:
        return lambda cls: register_node(node=cls, **meta)

    return register_node(node=node, **meta)


def find_global_node(node_id: str) -> Union[Behaviour, None]:
    """
    从全局查找节点
    :param node_id:
    :return:
    """
    if node_id in _ID_TO_NODES:
        return _ID_TO_NODES[node_id]
    return None


def add_global_node(node: Behaviour):
    """
    添加全局节点
    :param node:
    :return:
    """
    _ID_TO_NODES[node.id] = node


def print_all_node_cls():
    for name in _TAG_TO_NODES_CLS:
        print(name)


def find_node_cls(tag: str, allow_not_found: bool = True) -> Union[type, None]:
    if tag in _TAG_TO_NODES_CLS:
        return _TAG_TO_NODES_CLS[tag]
    elif tag in _CLASSNAME_TO_NODES_CLS:
        return _CLASSNAME_TO_NODES_CLS[tag]
    else:
        from .nodes import NotFound
        return NotFound
