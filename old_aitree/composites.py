from __future__ import annotations
from .behaviour import *
from .colors import *
from abc import ABC, abstractmethod
from . import utilities
import itertools


@register(type=NODE_TYPE.COMPOSITE, visible=False)
class Composite(Behaviour, ABC):
    """
    组合节点
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_child: typing.Optional[Behaviour] = None

    def reset(self, context: Context = None):
        super().reset(context=context)
        self.current_child = None

    @abstractmethod
    def tick(self) -> NODE_STATUS:
        pass

    def validate(self):
        """
        组合节点的所有子节点都必须是行为节点/条件节点/组合节点/装饰节点中的其中一个
        :return:
        """
        super().validate()
        for child in self.children:
            if child.meta['type'] not in [
                NODE_TYPE.ACTION,
                NODE_TYPE.CONDITION,
                NODE_TYPE.COMPOSITE,
                NODE_TYPE.DECORATOR,
                NODE_TYPE.SUB_TREE]:
                raise ValidateError(
                        "CompositeNode's child must be ActionNode/ConditionNode/CompositeNode/DecoratorNode/SubTree")
            child.validate()

    def stop(self, new_status: NODE_STATUS) -> None:
        if new_status == INVALID:
            self.current_child = None
            for child in self.children:
                if (
                        child.status != INVALID
                ):  # redundant if INVALID->INVALID
                    child.stop(new_status)

        # Regular Behaviour.stop() handling
        #   could call directly, but replicating here to avoid repeating the logger
        self.terminate(new_status)
        self.status = new_status


@register(tag=['Selector', 'Or'], order=2, color=COLORS.DEFAULT_SELECTOR)
class Selector(Composite):
    """
    Selectors are the decision makers.

    .. graphviz:: dot/selector.dot

    A selector executes each of its child behaviours in turn until one of them
    succeeds (at which point it itself returns :data:`~py_trees.common.Status.RUNNING`
    or :data:`~py_trees.common.Status.SUCCESS`,
    or it runs out of children at which point it itself returns :data:`~py_trees.common.Status.FAILURE`.
    We usually refer to selecting children as a means of *choosing between priorities*.
    Each child and its subtree represent a decreasingly lower priority path.

    .. note::

       Switching from a low -> high priority branch causes a `stop(INVALID)` signal to be sent to the previously
       executing low priority branch. This signal will percolate down that child's own subtree. Behaviours
       should make sure that they catch this and *destruct* appropriately.

    .. note::

       If configured with `memory`, higher priority checks will be skipped when a child returned with
       running on the previous tick. i.e. once a priority is locked in, it will run to completion and can
       only be interrupted if the selector is interrupted by higher priorities elsewhere in the tree.

    .. seealso:: The :ref:`py-trees-demo-selector-program` program demos higher priority switching under a selector.

    Args:
        memory (:obj:`bool`): if :data:`~py_trees.common.Status.RUNNING` on the previous tick,
            resume with the :data:`~py_trees.common.Status.RUNNING` child
        name (:obj:`str`): the composite behaviour name
        children ([:class:`~py_trees.behaviour.Behaviour`]): list of children to add
    """

    def __init__(
            self,
            *args, **kwargs
    ):
        super(Selector, self).__init__(*args, **kwargs)
        self.memory = self.get_prop('memory')

    def tick(self) -> typing.Iterator[Behaviour]:
        """
        Customise the tick behaviour for a selector.

        This implements priority-interrupt style handling amongst the selector's children.
        The selector's status is always a reflection of it's children's status.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # initialise
        if self.status != RUNNING:
            # selector specific initialisation - leave initialise() free for users to
            # re-implement without having to make calls to super()
            self.logger.debug(
                    "%s.tick() [!RUNNING->reset current_child]" % self.__class__.__name__
            )
            self.current_child = self.children[0] if self.children else None

            # reset the children - don't need to worry since they will be handled
            # a) prior to a remembered starting point, or
            # b) invalidated by a higher level priority

            # user specific initialisation
            self.initialise()

        # nothing to do
        if not self.children:
            self.current_child = None
            self.stop(FAILURE)
            yield self
            return

        # starting point
        if self.memory:
            assert self.current_child is not None  # should never be true, help mypy out
            index = self.children.index(self.current_child)
            # clear out preceding status' - not actually necessary but helps
            # visualise the case of memory vs no memory
            for child in itertools.islice(self.children, None, index):
                child.stop(INVALID)
        else:
            index = 0

        # actual work
        previous = self.current_child
        for child in itertools.islice(self.children, index, None):
            for node in child.tick():
                yield node
                if node is child:
                    if (
                            node.status == RUNNING
                            or node.status == SUCCESS
                    ):
                        self.current_child = child
                        self.status = node.status
                        if previous is None or previous != self.current_child:
                            # we interrupted, invalidate everything at a lower priority
                            passed = False
                            for child in self.children:
                                if passed:
                                    if child.status != INVALID:
                                        child.stop(INVALID)
                                passed = True if child == self.current_child else passed
                        yield self
                        return
        # all children failed, set failure ourselves and current child to the last bugger who failed us
        self.status = FAILURE
        try:
            self.current_child = self.children[-1]
        except IndexError:
            self.current_child = None
        yield self

    def stop(self, new_status: NODE_STATUS = INVALID) -> None:
        """
        Ensure that children are appropriately stopped and update status.

        Args:
            new_status : the composite is transitioning to this new status
        """
        self.logger.debug(
                f"{self.__class__.__name__}.stop()[{self.status}->{new_status}]"
        )
        Composite.stop(self, new_status)

    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        如果有一个节点分数是1，则返回1，否则返回所有节点分数的平均值
        :return:
        """
        if len(self.children) == 0:
            return 0
        score = 0
        for child in self.children:
            child_score = child.effect_score()
            if child_score == 1:
                return child_score
            score += child_score

        return score / len(self.children)


@register(tag=['Sequence', 'And'], order=1, color=COLORS.DEFAULT_SEQUENCE, props=[
    {
        'name'    : 'memory',
        'type'    : 'bool',
        'default' : True,
        'required': False,
        'desc'    : '记忆'
    },
])
class Sequence(Composite):
    """
    Sequences are the factory lines of behaviour trees.

    .. graphviz:: dot/sequence.dot

    A sequence will progressively tick over each of its children so long as
    each child returns :data:`~py_trees.common.Status.SUCCESS`. If any child returns
    :data:`~py_trees.common.Status.FAILURE` or :data:`~py_trees.common.Status.RUNNING`
    the sequence will halt and the parent will adopt
    the result of this child. If it reaches the last child, it returns with
    that result regardless.

    .. note::

       The sequence halts once it engages with a child is RUNNING, remaining behaviours
       are not ticked.

    .. note::

       If configured with `memory` and a child returned with running on the previous tick, it will
       proceed directly to the running behaviour, skipping any and all preceding behaviours. With memory
       is useful for moving through a long running series of tasks. Without memory is useful if you
       want conditional guards in place preceding the work that you always want checked off.

    .. seealso:: The :ref:`py-trees-demo-sequence-program` program demos a simple sequence in action.

    Args:
        name: the composite behaviour name
        memory: if :data:`~py_trees.common.Status.RUNNING` on the previous tick,
            resume with the :data:`~py_trees.common.Status.RUNNING` child
        children: list of children to add
    """

    def __init__(
            self,
            *args, **kwargs
    ):
        super(Sequence, self).__init__(*args, **kwargs)
        self.memory = self.get_prop('memory')

    def tick(self) -> typing.Iterator[Behaviour]:
        """
        Tick over the children.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)

        # initialise
        index = 0
        if self.status != RUNNING:
            self.current_child = self.children[0] if self.children else None
            for child in self.children:
                if child.status != INVALID:
                    child.stop(INVALID)
            self.initialise()  # user specific initialisation
        elif self.memory and RUNNING:
            assert self.current_child is not None  # should never be true, help mypy out
            index = self.children.index(self.current_child)
        elif not self.memory and RUNNING:
            self.current_child = self.children[0] if self.children else None
        else:
            # previous conditional checks should cover all variations
            raise RuntimeError("Sequence reached an unknown / invalid state")

        # nothing to do
        if not self.children:
            self.current_child = None
            self.stop(SUCCESS)
            yield self
            return

        # actual work
        for child in itertools.islice(self.children, index, None):
            for node in child.tick():
                yield node
                if node is child and node.status != SUCCESS:
                    self.status = node.status
                    if not self.memory:
                        # invalidate the remainder of the sequence
                        # i.e. kill dangling runners
                        for child in itertools.islice(self.children, index + 1, None):
                            if child.status != INVALID:
                                child.stop(INVALID)
                    yield self
                    return
            try:
                # advance if there is 'next' sibling
                self.current_child = self.children[index + 1]
                index += 1
            except IndexError:
                pass

        self.stop(SUCCESS)
        yield self

    def stop(self, new_status: NODE_STATUS = INVALID) -> None:
        """
        Ensure that children are appropriately stopped and update status.

        Args:
            new_status : the composite is transitioning to this new status
        """
        self.logger.debug(
                f"{self.__class__.__name__}.stop()[{self.status}->{new_status}]"
        )
        Composite.stop(self, new_status)

    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        所有节点分数的平均值
        :return:
        """
        # 只有所有子节点都成功，才算成功
        if len(self.children) == 0:
            return 0
        score = 0
        for child in self.children:
            score += child.effect_score()
        return score / len(self.children)


Or = Selector
And = Sequence


@register(
        props={
            'success_threshold': {
                'type'   : 'int',
                'default': 1,
                'desc'   : '成功阈值'
            },
            'synchronise'      : {
                'type'   : 'bool',
                'default': True,
                'desc'   : 'stop ticking of children with status SUCCESS until the success_threshold criteria is met'
            }
        },
        order=3,
        color=COLORS.DEFAULT_PARALLEL
)
class Parallel(Composite):
    """
    Parallels enable a kind of spooky at-a-distance concurrency.

    .. graphviz:: dot/parallel.dot

    A parallel ticks every child every time the parallel is itself ticked.
    The parallelism however, is merely conceptual. The children have actually been
    sequentially ticked, but from both the tree and the parallel's purview, all
    children have been ticked at once.

    The parallelism too, is not true in the sense that it kicks off multiple threads
    or processes to do work. Some behaviours *may* kick off threads or processes
    in the background, or connect to existing threads/processes. The behaviour itself
    however, merely monitors these and is itself encosced in a py_tree which only ever
    ticks in a single-threaded operation.

    * Parallels will return :data:`~py_trees.common.Status.FAILURE` if any
      child returns :py:data:`~py_trees.common.Status.FAILURE`
    * Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnAll`
      only returns :py:data:`~py_trees.common.Status.SUCCESS` if **all** children
      return :py:data:`~py_trees.common.Status.SUCCESS`
    * Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnOne`
      return :py:data:`~py_trees.common.Status.SUCCESS` if **at least one** child
      returns :py:data:`~py_trees.common.Status.SUCCESS` and others are
      :py:data:`~py_trees.common.Status.RUNNING`
    * Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnSelected`
      only returns :py:data:`~py_trees.common.Status.SUCCESS` if a **specified subset**
      of children return :py:data:`~py_trees.common.Status.SUCCESS`

    Policies :class:`~py_trees.common.ParallelPolicy.SuccessOnAll` and
    :class:`~py_trees.common.ParallelPolicy.SuccessOnSelected` may be configured to be
    *synchronised* in which case children that tick with
    :data:`~py_trees.common.Status.SUCCESS` will be skipped on subsequent ticks until
    the policy criteria is met, or one of the children returns
    status :data:`~py_trees.common.Status.FAILURE`.

    Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnSelected` will
    check in both the :meth:`~py_trees.behaviour.Behaviour.setup` and
    :meth:`~py_trees.behaviour.Behaviour.tick` methods to to verify the
    selected set of children is actually a subset of the children of this parallel.

    .. seealso::
       * :ref:`Context Switching Demo <py-trees-demo-context-switching-program>`
    """

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        """
        Initialise the behaviour with name, policy and a list of children.

        Args:
            name: the composite behaviour name
            policy: policy for deciding success or otherwise (default: SuccessOnAll)
            children: list of children to add
        """
        super(Parallel, self).__init__(*args, **kwargs)
        self.success_threshold = self.get_prop('success_threshold')
        self.synchronise = self.get_prop('synchronise')

    def tick(self) -> typing.Iterator[Behaviour]:
        """
        Tick over the children.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children

        Raises:
            RuntimeError: if the policy configuration was invalid
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)

        # reset
        if self.status != RUNNING:
            self.logger.debug("%s.tick(): re-initialising" % self.__class__.__name__)
            for child in self.children:
                # reset the children, this ensures old SUCCESS/FAILURE status flags
                # don't break the synchronisation logic below
                if child.status != INVALID:
                    child.stop(INVALID)
            self.current_child = None
            # subclass (user) handling
            self.initialise()

        # nothing to do
        if not self.children:
            self.current_child = None
            self.stop(SUCCESS)
            yield self
            return

        # process them all first
        for child in self.children:
            if self.synchronise and child.status == SUCCESS:
                continue
            for node in child.tick():
                yield node

        # determine new status
        new_status = RUNNING
        self.current_child = self.children[-1]
        try:
            failed_child = next(
                    child
                    for child in self.children
                    if child.status == FAILURE
            )
            self.current_child = failed_child
            new_status = FAILURE
        except StopIteration:
            if self.success_threshold == -1:
                if all([c.status == SUCCESS for c in self.children]):
                    new_status = SUCCESS
                    self.current_child = self.children[-1]
            else:
                successful = [
                    child
                    for child in self.children
                    if child.status == SUCCESS
                ]
                if len(successful) >= self.success_threshold:
                    new_status = SUCCESS
                    self.current_child = successful[-1]
        # this parallel may have children that are still running
        # so if the parallel itself has reached a final status, then
        # these running children need to be terminated so they don't dangle
        if new_status != RUNNING:
            self.stop(new_status)
        self.status = new_status
        yield self

    def stop(self, new_status=INVALID) -> None:
        """
        Ensure that any running children are stopped.

        Args:
            new_status : the composite is transitioning to this new status
        """
        self.logger.debug(
                f"{self.__class__.__name__}.stop()[{self.status}->{new_status}]"
        )

        # clean up dangling (running) children
        for child in self.children:
            if child.status == RUNNING:
                # this unfortunately knocks out it's running status for introspection
                # but logically is the correct thing to do, see #132.
                child.stop(INVALID)
        Composite.stop(self, new_status)

    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        子节点分数从大到小排序，取第success_threshold个
        :return:
        """
        if len(self.children) == 0:
            return 0

        # 只要有一个子节点成功，就算成功
        scores = [s for s in [child.effect_score() for child in self.children]]
        # 从大到小排序
        scores.sort(reverse=True)

        # 如果第success_threshold个节点分数为1，则返回1
        if scores[self.success_threshold - 1] == 1:
            return 1
        # 返回平均值
        return sum(scores) / len(scores)

# @register(color=COLORS.DEFAULT_RANDOM)
# class RandomSelector(Composite):
#     """
#     每次随机一个未执行的节点，总随机次数为子节点个数
#     - 当前执行节点返回 SUCCESS，退出停止，向父节点返回 SUCCESS
#     - 当前执行节点返回 FAILURE，退出当前节点，继续随机一个未执行的节点开始执行
#     - 当前执行节点返回 Running，记录当前节点，向父节点返回 Running，下次执行直接从该节点开始
#     - 如果所有节点都返回FAILURE，执行完所有节点后，向父节点返回 FAILURE
#     """
#
#     def execute(self) -> NODE_STATUS:
#         if len(self.children) == 0:
#             # 没有子节点，直接返回失败
#             return FAILURE
#
#         if len(self.running_indexes) > 0:
#             # 如果当前存在正在运行列表，则从正在运行列表的第一个节点开始执行
#             select_index = self.running_indexes[0]
#             status = INVALID
#             for index, child in enumerate(self.children):
#                 if index != select_index:
#                     # 如果当前存在正在运行列表，则继续执行正在运行的节点
#                     child.skip()
#                     continue
#                 status = child.tick()
#                 if status == RUNNING:
#                     self.running_indexes = [index]
#             return status
#
#         import random
#         # shuffle
#         indexes = list(range(len(self.children)))
#         random.shuffle(indexes)
#
#         status = INVALID
#
#         for index in indexes:
#             child = self.children[index]
#             if status == SUCCESS or status == RUNNING:
#                 # 跳过剩余的节点
#                 child.skip()
#                 continue
#
#             status = child.tick()
#             if status == RUNNING:
#                 self.running_indexes = [index]
#
#         return status


# # register
# class RandomSequence(CompositeNode):
#     """
#     随机顺序执行节点
#     """
#
#     def execute(self) -> NODE_STATUS:
#         if len(self.children) == 0:
#             # 没有子节点，直接返回失败
#             return FAILURE
#
#         if len(self.running_indexes) > 0:
#             # 如果当前存在正在运行列表，则从正在运行列表的第一个节点开始执行
#             select_index = self.running_indexes[0]
#             status = INVALID
#             for index, child in enumerate(self.children):
#                 if index != select_index:
#                     # 如果当前存在正在运行列表，则继续执行正在运行的节点
#                     child.skip()
#                     continue
#                 status = child.tick()
#                 if status == RUNNING:
#                     self.running_indexes = [index]
#             return status
#
#         import random
#         # shuffle
#         indexes = list(range(len(self.children)))
#         random.shuffle(indexes)
#
#         status = INVALID
#
#         for index in indexes:
#             child = self.children[index]
#             if status == FAILURE or status == RUNNING:
#                 # 跳过剩余的节点
#                 child.skip()
#                 continue
#
#             status = child.tick()
#             if status == RUNNING:
#                 self.running_indexes = [index]
#
#         return status
