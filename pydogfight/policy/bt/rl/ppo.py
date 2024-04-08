from __future__ import annotations

import pybts
from py_trees.behaviour import Behaviour

from pydogfight.policy.bt.nodes import *

from typing import Any, SupportsFloat

from gymnasium.core import ActType, ObsType
from py_trees.common import Status
from pybts.composites import Composite, Selector, Sequence
from stable_baselines3 import PPO
import py_trees
import gymnasium as gym
from pybts.rl import RLOnPolicyNode
import typing
import jinja2
from pydogfight.core.actions import Actions


class PPONode(BTPolicyNode, RLOnPolicyNode, ABC):

    def __init__(self,
                 path: str,
                 policy: str = 'MlpPolicy',
                 save_interval: int = 10,
                 tensorboard_log: typing.Optional[str] = None,
                 log_interval: int = 1,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        RLOnPolicyNode.__init__(self)
        self.path = path
        self.policy = policy
        self.save_interval = save_interval
        self.tensorboard_log = tensorboard_log
        self.log_interval = log_interval

    @classmethod
    def creator(cls, d: dict, c: list):
        return cls(**d)

    def to_data(self):
        return {
            **super().to_data(),
            **RLOnPolicyNode.to_data(self),
            'policy': self.policy,
            'path'  : self.path,
        }

    def setup(self, **kwargs: typing.Any) -> None:
        self.path = jinja2.Template(self.path).render({
            'color'     : self.agent.color,
            'agent_name': self.agent_name
        })
        if self.tensorboard_log is not None:
            self.tensorboard_log = jinja2.Template(self.tensorboard_log).render({
                'color'     : self.agent.color,
                'agent_name': self.agent_name
            })

        self.rl_ppo_setup_model(
                train=self.env.options.train,
                path=self.path,
                policy=self.policy,
                tensorboard_log=self.tensorboard_log,
                verbose=1,
                # n_steps=8,
                # batch_size=8,
                tb_log_name=self.agent_name
        )

    def rl_env(self) -> gym.Env:
        return self.env

    def rl_action_space(self) -> gym.spaces.Space:
        return self.env.agent_action_space

    def rl_observation_space(self) -> gym.spaces.Space:
        return self.env.agent_observation_space

    def rl_gen_obs(self):
        return self.env.gen_agent_obs(self.agent_name)

    def rl_gen_info(self) -> dict:
        return self.env.gen_info()

    def rl_gen_reward(self) -> float:
        return self.env.gen_reward(color=self.agent.color, previous=self.rl_info) + self.rl_gen_status_reward()

    def rl_gen_status_reward(self) -> float:
        # 生成节点运行状态的奖励
        status_reward = self.env.options.status_reward[self.status.name] * max(
                self.env.options.update_interval,
                self.env.options.policy_interval)
        return status_reward

    def reset(self):
        super().reset()
        if self.env.options.train:
            self.rl_model.logger.record("reward/round", self.rl_reward)
            self.rl_model.logger.record("wins", self.env.game_info[f'{self.agent.color}_wins'])
            self.rl_model.logger.record("loses", self.env.game_info[f'{self.agent.enemy_color}_wins'])
            self.rl_model.logger.record("draws", self.env.game_info['draws'])
        RLOnPolicyNode.reset(self)


class PPOComposite(PPONode, Composite):
    def __init__(self,
                 children: typing.Optional[typing.List[py_trees.behaviour.Behaviour]] = None,
                 **kwargs
                 ):
        PPONode.__init__(
                self,
                **kwargs
        )
        Composite.__init__(self, children=children, **kwargs)

    @classmethod
    def creator(cls, d: dict, c: list):
        return cls(
                **d,
                children=c,
        )


class PPOSwitcher(PPOComposite):
    """
    选择其中一个子节点来执行
    """

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.children))

    def switcher_tick(
            self,
            tick_again_status: list[Status],
    ):
        """Sequence/Selector的tick逻辑"""
        self.debug_info['tick_count'] += 1
        self.logger.debug("%s.tick()" % (self.__class__.__name__))

        if self.status in tick_again_status:
            # 重新执行上次执行的子节点
            assert self.current_child is not None
        else:
            index = self.rl_take_action(
                    train=self.env.options.train,
                    log_interval=self.log_interval,
                    save_interval=self.save_interval,
                    save_path=self.path
            )
            self.current_child = self.children[index]
        yield from self.current_child.tick()

        # 剩余的子节点全部停止
        for child in self.children:
            if child == self.current_child:
                continue
            # 清除子节点的状态（停止正在执行的子节点）
            child.stop(Status.INVALID)

        self.status = self.current_child.status
        yield self

    def tick(self) -> typing.Iterator[Behaviour]:
        yield from self.switcher_tick(tick_again_status=[Status.RUNNING])


class ReactivePPOSwitcher(PPOSwitcher):
    """
    每次都会重新开始
    """

    def tick(self) -> typing.Iterator[Behaviour]:
        yield from self.switcher_tick(tick_again_status=[])


class PPOSelector(PPOComposite, Selector):
    """
    将某个子节点作为开头，重置children并执行（事后回还原）
    self.children = self.children[index:] + self.children[:index]
    """
    init_children: list[py_trees.behaviour.Behaviour]

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.children))

    def setup(self, **kwargs: typing.Any) -> None:
        self.init_children = self.children[:]
        super().setup(**kwargs)

    def tick(self) -> typing.Iterator[Behaviour]:
        index = self.rl_take_action(
                train=self.env.options.train,
                log_interval=self.log_interval,
                save_interval=self.save_interval,
                save_path=self.path
        ) or 0
        self.children = self.init_children[index:] + self.init_children[:index]
        yield from Selector.tick(self)


class PPOCondition(PPONode, pybts.Condition):
    """
    条件节点：用PPO来判断是否需要执行
    """

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(2)

    def update(self) -> Status:
        action = self.rl_take_action(
                train=self.env.options.train,
                log_interval=self.log_interval,
                save_interval=self.save_interval,
                save_path=self.path
        ) or 0
        if action == 0:
            return Status.FAILURE
        else:
            return Status.SUCCESS


class PPOAction(PPONode):
    def __init__(
            self,
            allow_actions: str = 'keep,go_to_location,fire_missile,go_home',
            **kwargs
    ):
        super().__init__(**kwargs)
        if allow_actions == '':
            allow_actions = 'keep,go_to_location,fire_missile,go_home'
        self.allow_actions = allow_actions.split(',')

    @classmethod
    def creator(cls, d: dict, c: list):
        return cls(
                **d,
        )

    def to_data(self):
        return {
            **super().to_data(),
            **RLOnPolicyNode.to_data(self),
            'allow_actions': self.allow_actions
        }

    # def rl_action_space(self) -> gym.spaces.Space:
    #     return self.env.agent_action_space

    def update(self) -> Status:
        action = self.rl_take_action(
                train=self.env.options.train,
                log_interval=self.log_interval,
                save_interval=self.save_interval,
                save_path=self.path
        )

        relative_wpt = self.agent.waypoint.relative_move(
                dx=action[1] * self.agent.radar_radius,
                dy=action[2] * self.agent.radar_radius)

        allow_actions = list(map(lambda x: Actions.from_str(x), self.allow_actions))

        action_type = Actions.extract_action_in_value_range(actions=allow_actions, value=action[0], value_range=(-1, 1))
        self.actions.put_nowait((action_type, relative_wpt.x, relative_wpt.y))
        return Status.SUCCESS

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(3,))


class PPOActionPPA(PPOComposite, PPOAction):
    """
    执行反向链式动作

    只有一个子节点：
    - 后置条件：
        不满足：奖励为0
        满足：奖励为1

    有两个子节点：
    - 前置条件：
        不满足：奖励为-1
        满足：奖励为为0
    - 后置条件：同上

    后置条件优先级大于前置条件
    状态：
    - 后置条件满足：SUCCESS
    - 前置条件满足：RUNNING
    - 否则：FAILURE
    """

    def __init__(self, **kwargs):
        PPOAction.__init__(self, **kwargs)
        PPOComposite.__init__(self, **kwargs)

    @classmethod
    def creator(cls, d: dict, c: list):
        return cls(
                children=c,
                **d,
        )

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        assert len(self.children) in [1, 2], f'PPOActionPPA: children count ({len(self.children)}) must be 1 or 2'

    def post_condition(self) -> pybts.Node | None:
        assert len(self.children) in [1, 2]
        return self.children[-1]

    def pre_condition(self) -> pybts.Node | None:
        if len(self.children) == 2:
            return self.children[0]
        return None

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(3,))

    def rl_gen_reward(self) -> float:
        pre_c = self.pre_condition()
        post_c = self.post_condition()

        if post_c is not None:
            post_c.tick_once()
            if post_c.status == Status.SUCCESS:
                return 1

        if pre_c is not None:
            pre_c.tick_once()
            if pre_c.status != Status.SUCCESS:
                return -1

        return 0

    def update(self) -> Status:
        PPOAction.update(self)

        pre_c = self.pre_condition()
        post_c = self.post_condition()

        if post_c is not None:
            post_c.tick_once()
            if post_c.status == Status.SUCCESS:
                return Status.SUCCESS
        if pre_c is not None:
            pre_c.tick_once()
            if pre_c.status == Status.SUCCESS:
                return Status.RUNNING
        return Status.FAILURE
