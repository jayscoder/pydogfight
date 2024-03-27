from __future__ import annotations
from .register import register, add_global_node
from .config import *
import copy
from .context import Context
from . import logging, utilities
import typing
import xml.etree.ElementTree as ET
from abc import ABC


@register(tag=['root'], type=NODE_TYPE.VIRTUAL, props={ }, visible=False)
class Behaviour(ABC):
    meta = copy.deepcopy(META_TEMPLATE)  # 节点类元数据，用于存储一些额外的信息

    def __init__(self, *args, **kwargs):
        self.children: List[Behaviour] = []
        self.parent: Behaviour | None = None
        self._context = kwargs.get('context', Context())  # 节点上下文
        assert isinstance(self._context, Context)
        self.meta = copy.deepcopy(self.__class__.meta)
        for k in kwargs:
            if k in META_TEMPLATE:
                self.meta[k] = kwargs[k]
        self.attributes = copy.deepcopy(kwargs)
        self.id = self.get_prop('id', default=utilities.new_node_id())

        self.cache = { }  # 用于缓存一些节点数据，只有在reset的时候才会清空
        self.status: NODE_STATUS = NODE_STATUS(value=kwargs.get('status', 'not_run'))
        self.logger = logging.Logger(self.label)
        self.feedback_message = ""  # useful for debugging, or human readable updates, but not necessary to implement

        add_global_node(self)

        for arg in args:
            if isinstance(arg, Behaviour):
                self.add_child(arg)
            elif callable(arg):
                self.add_child(arg())
            # elif isinstance(arg, NODE_STATUS):
            #     from .decorators import ForceSuccess, ForceFail, ForceRunning, ForceNotRun
            #     if arg == SUCCESS:
            #         self.add_child(ForceSuccess())
            #     elif arg == FAILURE:
            #         self.add_child(ForceFail())
            #     elif arg == RUNNING:
            #         self.add_child(ForceRunning())
            #     else:
            #         self.add_child(ForceNotRun())

    def __call__(self, *args, **kwargs):
        # 用于复制节点
        node = self.__class__(*args, **self.attributes, **kwargs)
        node.context = self.context
        node.meta = copy.deepcopy(self.meta)
        # 复制所有子节点
        for child in self.children:
            node.add_child(child.__call__())
        return node

    @property
    def label(self):
        """
        节点的标签，用于显示在webui上的节点名称
        """
        return self.get_prop('label') or self.meta.get('label', '')

    @label.setter
    def label(self, value: str):
        self.attributes['label'] = value

    @property
    def desc(self):
        return self.get_prop('desc') or self.meta.get('desc', '')

    @desc.setter
    def desc(self, value: str):
        self.attributes['desc'] = value

    @property
    def disabled(self):
        return self.meta.get('disabled', True)

    @disabled.setter
    def disabled(self, value):
        self.meta['disabled'] = value

    @property
    def visible(self):
        """
        节点是否可见（在webui上）
        """
        return self.meta.get('visible', True)

    @visible.setter
    def visible(self, value):
        """
        设置节点是否可见（在webui上）
        """
        self.meta['visible'] = value

    @property
    def sorted_children(self):
        # 返回子节点列表排序后的列表
        return sorted(self.children, key=lambda x: x.__class__.__name__, reverse=True)

    def add_child(self, *child):
        for ch in child:
            if not isinstance(ch, Behaviour):
                raise Exception(f'child must be Node, but got {type(ch)}')
            ch.context = self.context
            ch.parent = self
            self.children.append(ch)
        return self

    def find_node(self, by) -> Union['Behaviour', None]:
        if isinstance(by, str):
            if by == self.id:
                return self
            for child in self.children:
                node = child.find_node(by)
                if node is not None:
                    return node
        elif isinstance(by, type):
            if isinstance(self, by):
                return self
            for child in self.children:
                node = child.find_node(by)
                if node is not None:
                    return node
        elif isinstance(by, list):
            for i_by in by:
                node = self.find_node(i_by)
                if node is not None:
                    return node
        elif isinstance(by, Behaviour):
            return self.find_node(by.id)
        elif callable(by):
            if by(self):
                return self
            for child in self.children:
                node = child.find_node(by)
                if node is not None:
                    return node
        return None

    def find_nodes(self, by) -> List['Behaviour']:
        nodes = []
        if isinstance(by, str):
            if by == self.id:
                nodes.append(self)
            for child in self.children:
                nodes.extend(child.find_nodes(by))
        elif isinstance(by, Behaviour):
            nodes.extend(self.find_nodes(by.id))
        elif isinstance(by, list):
            for i_by in by:
                nodes.extend(self.find_nodes(i_by))
        elif isinstance(by, type):
            if isinstance(self, by):
                nodes.append(self)
            for child in self.children:
                nodes.extend(child.find_nodes(by))
        elif callable(by):
            if by(self):
                nodes.append(self)
            for child in self.children:
                nodes.extend(child.find_nodes(by))
        return nodes

    @property
    def root(self):
        if self.parent is None:
            return self
        return self.parent.root

    @property
    def depth(self):
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value
        for child in self.children:
            child.context = value

    def reset(self, context: Context = None):
        if context is not None:
            self._context = context
        self.status = INVALID
        self.cache = { }
        for child in self.children:
            child.reset(context=self._context)

    def skip(self):
        """
        跳过当前节点，当前节点不执行
        :return:
        """
        self.status = INVALID
        for child in self.children:
            child.skip()

    def condition(self) -> Union[Behaviour, None]:
        """
        行为树原语：前置条件，用于判断当前节点的执行条件
        """
        return None

    def effect(self) -> Union[Behaviour, None]:
        """
        行为树原语：预期执行效果，用于判断当前节点的目标是否达成
        :return:
        """
        return None

    # 度量分数
    def effect_score(self) -> float:
        """
        行为树原语：当前节点的度量分数
        :return:
        """
        eff = self.effect()
        if eff is None:
            return self.status.score
        eff.parent = self
        eff.context = self.context
        return eff.effect_score()

    @classmethod
    def module_name(cls):
        return f'{cls.__module__}.{cls.__qualname__}'

    @property
    def blackboard(self):
        return self.context.blackboard

    def validate(self):
        """
        验证节点是否合法
        :return: (是否合法, 错误信息)
        """
        if self.id == '' or self.id is None:
            raise ValidateError("id can not be empty")

        # 校验参数
        for prop in self.meta['props']:
            if prop['required']:
                if prop['name'] not in self.attributes:
                    raise ValidateError("prop {} is required".format(prop['name']))
            if prop['type'] != str:
                # 如果prop的类型不是str，则需要校验类型
                value = self.get_prop(prop['name'])
                if not isinstance(value, eval(prop['type'])):
                    raise ValidateError(
                            "prop {} type error, expect {}, but got {}".format(prop['name'], prop['type'], type(value)))
        if self.depth > MAX_TREE_DEPTH:
            raise ValidateError(f'tree depth too large, max depth is {MAX_TREE_DEPTH}')

        # 校验子节点
        for child in self.children:
            child.validate()

    def tick_once(self):
        for _ in self.tick():
            pass

    def tick(self) -> typing.Iterator[Behaviour]:
        """
        执行节点
        :return:
        """
        self.logger.debug("%s.tick()" % self.label)

        if 'disabled' in self.meta and self.meta['disabled']:
            self.skip()
            self.status = INVALID
            yield self
            return

        if self.status != RUNNING:
            self.initialise()

        # don't set self.status yet, terminate() may need to check what the current state is first
        new_status = self.execute()

        if new_status is None:
            # 没有返回值的话，视为执行成功
            new_status = SUCCESS

        if new_status != RUNNING:
            self.stop(new_status)

        self.status = new_status
        yield self



    def initialise(self) -> None:  # noqa: B027
        """
        Execute user specified instructions prior to commencement of a new round of activity.

        Users should override this method to perform any necessary initialising/clearing/resetting
        of variables prior to a new round of activity for the behaviour.

        This method is automatically called via the :meth:`py_trees.behaviour.Behaviour.tick` method
        whenever the behaviour is not :data:`~py_trees.common.Status.RUNNING`.

        ... note:: This method can be called more than once in the lifetime of a tree!
        """
        pass

    def execute(self) -> NODE_STATUS:
        """
        具体的执行行为，用户不应该直接调用这个方法，而是调用tick方法
        用户应该重写这个方法
        :return:
        """
        pass

    def stop(self, new_status: NODE_STATUS) -> None:
        """
        Stop the behaviour with the specified status.

        Args:
            new_status: the behaviour is transitioning to this new status

        This is called to bring the current round of activity for the behaviour to completion, typically
        resulting in a final status of :data:`~py_trees.common.Status.SUCCESS`,
        :data:`~py_trees.common.Status.FAILURE` or :data:`~py_trees.common.Status.INVALID`.

        .. warning::
           Users should not override this method to provide custom termination behaviour. The
           :meth:`~py_trees.behaviour.Behaviour.terminate` method has been provided for that purpose.
        """
        self.logger.debug(
                "%s.stop(%s)"
                % (
                    self.__class__.__name__,
                    "%s->%s" % (self.status, new_status)
                    if self.status != new_status
                    else "%s" % new_status,
                )
        )
        self.terminate(new_status)
        self.status = new_status

    def terminate(self, new_status: NODE_STATUS) -> None:  # noqa: B027
        """
        Execute user specified instructions when the behaviour is stopped.

        Users should override this method to clean up.
        It will be triggered when a behaviour either
        finishes execution (switching from :data:`~py_trees.common.Status.RUNNING`
        to :data:`~py_trees.common.Status.FAILURE` || :data:`~py_trees.common.Status.SUCCESS`)
        or it got interrupted by a higher priority branch (switching to
        :data:`~py_trees.common.Status.INVALID`). Remember that
        the :meth:`~py_trees.behaviour.Behaviour.initialise` method
        will handle resetting of variables before re-entry, so this method is about
        disabling resources until this behaviour's next tick. This could be a indeterminably
        long time. e.g.

        * cancel an external action that got started
        * shut down any temporary communication handles

        Args:
            new_status (:class:`~py_trees.common.Status`): the behaviour is transitioning to this new status

        .. warning:: Do not set `self.status = new_status` here, that is automatically handled
           by the :meth:`~py_trees.behaviour.Behaviour.stop` method.
           Use the argument purely for introspection purposes (e.g.
           comparing the current state in `self.status` with the state it will transition to in
           `new_status`.

        .. seealso:: :meth:`py_trees.behaviour.Behaviour.stop`
        """
        pass

    def tip(self) -> typing.Optional[Behaviour]:
        """
        Get the *tip* of this behaviour's subtree (if it has one).

        This corresponds to the the deepest node that was running before the
        subtree traversal reversed direction and headed back to this node.

        Returns:
            The deepest node (behaviour) that was running before subtree traversal
            reversed direction, or None if this behaviour's status is
            :data:`~py_trees.common.Status.INVALID`.
        """
        return self if self.status != INVALID else None

    def iterate(self, direct_descendants: bool = False) -> typing.Iterator[Behaviour]:
        """
        Iterate over this child and it's children.

        This utilises python generators for looping. To traverse the entire tree:

        .. code-block:: python

           for node in my_behaviour.iterate():
               print("Name: {0}".format(node.name))

        Args:
            direct_descendants (:obj:`bool`): only yield children one step away from this behaviour.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: one of it's children
        """
        for child in self.children:
            if not direct_descendants:
                for node in child.iterate():
                    yield node
            else:
                yield child
        yield self

    def get_prop(self, name, default=None, once: bool = False):
        """
        获取节点参数
        - 如果参数值被{{}}包裹，则视为变量，需要从黑板中获取，支持jinja2模板语法
        - 否则从attributes中获取
        :param name:
        :param default:
        :param once: 是否只获取一次，如果为True，则只获取一次，之后直接存到cache里
        :return:
        """
        if once and name in self.cache:
            return self.cache[name]

        if isinstance(name, list):
            # 如果name是列表，则视为同一个参数可以有多种表示形式，优先使用靠前的
            for i_name in name:
                value = self.get_prop(name=i_name, default=default, once=once)
                if value is not None:
                    return value
            return None

        prop_rule = utilities.find_prop(self.meta, name=name)
        value = self.attributes.get(name, default)
        # if value is None:
        #     value = self.blackboard.get(name, default)

        if value is None and prop_rule is not None:
            value = prop_rule['default']

        if callable(value):
            value = value()

        if prop_rule is not None:
            value = utilities.parse_type_value(value, value_type=prop_rule['type'])

        if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
            from jinja2 import Template
            value = Template(value).render(self.blackboard)

        if prop_rule is not None:
            value = utilities.parse_type_value(value, value_type=prop_rule['type'])

        if once:
            self.cache[name] = value

        return value

    def to_xml_node(self, ignore_children: bool = False, sorted_children: bool = False, **kwargs):
        """
        将节点转换为xml节点
        :return:
        """
        attrs = copy.deepcopy(self.attributes)
        if kwargs.get('status', False):
            attrs['status'] = self.status.__str__()

        if kwargs.get('effect_score', False):
            attrs['effect_score'] = self.effect_score()

        if kwargs.get('meta', False):
            attrs = { **attrs, **self.meta }

        if kwargs.get('id', False):
            attrs['id'] = self.id

        if kwargs.get('props', False):
            props = self.meta['props']

            for prop in props:
                attrs[prop['name']] = self.get_prop(prop['name'])

        # if kwargs.get('seed', False):
        #     if self.simulation is not None:
        #         attrs['seed'] = self.simulation.seed
        #     kwargs.pop('seed')

        for key in attrs:
            attrs[key] = str(attrs[key])

        node = ET.Element(self.__class__.__name__, attrib=attrs)
        if not ignore_children:
            if sorted_children:
                children = self.sorted_children
            else:
                children = self.children
            for child in children:
                node.append(child.to_xml_node(
                        ignore_children=ignore_children,
                        sorted_children=sorted_children,
                        **kwargs))
        return node

    def to_xml(self, ignore_children: bool = False, sorted_children: bool = False, **kwargs):
        """
        将节点转换为xml字符串
        :return:
        """
        from xml.dom import minidom
        xml_node = self.to_xml_node(
                ignore_children=ignore_children,
                sorted_children=sorted_children,
                **kwargs)
        text = ET.tostring(xml_node, encoding='utf-8').decode('utf-8')
        text = minidom.parseString(text).toprettyxml(indent='    ').replace('<?xml version="1.0" ?>', '').strip()
        return text

    def __str__(self):
        return self.to_xml(ignore_children=True, id=True)

    def __repr__(self):
        return self.__str__()

    def tag(self, deep: bool = False):
        if 'tag' in self.meta:
            tag = self.meta['tag'][0]
        else:
            tag = self.__class__.__name__
        if deep:
            for child in self.children:
                tag += child.tag(deep=deep)
        return tag

    def to_json(self):
        """
        将节点转换为json
        :return:
        """
        node_tags = self.meta.get('tag', [])
        node_label = self.label or self.tag()
        json_data = {
            'id'      : self.id,
            'label'   : node_label,
            'children': [child.to_json() for child in self.children],
            'data'    : {
                'tag'   : node_tags,
                'key'   : node_label,
                'name'  : node_label,
                'label' : node_label,
                'desc'  : self.desc,
                'ref_id': self.get_prop('ref_id', ''),
                **self.meta,
                **self.attributes,
            }
        }

        json_data['data']['params'] = copy.deepcopy(json_data['data']['props'])
        for param in json_data['data']['params']:
            param['key'] = param['name']
            param['value'] = str(self.get_prop(param['name']))

        return json_data

    def node_count(self) -> int:
        """
        获取节点数量
        :return:
        """
        count = 1
        for child in self.children:
            count += child.node_count()
        return count

    def edge_count(self) -> int:
        """
        获取边数量
        :return:
        """
        count = len(self.children)
        for child in self.children:
            count += child.edge_count()
        return count
