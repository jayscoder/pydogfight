from __future__ import annotations

import pybts
from py_trees.behaviour import Behaviour
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from pydogfight.policy.bt.base_class import *

from typing import Any, SupportsFloat, Union

from gymnasium.core import ActType, ObsType
from py_trees.common import Status
from pybts.composites import Composite, Selector, Sequence
from stable_baselines3 import PPO, SAC, HER
import py_trees
import gymnasium as gym
from pybts.rl import RLBaseNode
import typing
import jinja2
from pydogfight.core.actions import Actions


class RLNode(BTPolicyNode, RLBaseNode, ABC):

    def __init__(self,
                 path: str = '',
                 algo: str | BaseAlgorithm = 'PPO',
                 reward_scope: str = '',
                 save_interval: int | str = 30,
                 tensorboard_log: typing.Optional[str] = None,
                 log_interval: int | str = 10,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        RLBaseNode.__init__(self)
        self.algo = algo
        self.reward_scope = reward_scope  # 如果scope设置成default或其他不为空的值，则认为奖励要从context.rl_reward[scope]中拿
        self.path = path
        self.tensorboard_log = tensorboard_log
        self.save_interval = save_interval
        self.log_interval = log_interval

    def to_data(self):
        return {
            **super().to_data(),
            **RLBaseNode.to_data(self),
            'algo'         : str(self.algo),
            'path'         : self.path,
            'reward_scope' : self.reward_scope,
            'save_interval': self.save_interval,
            'log_interval' : self.log_interval
        }

    def rl_policy(self) -> Union[str, typing.Type[ActorCriticPolicy]]:
        return ActorCriticPolicy

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        if isinstance(self.algo, str):
            print(self.context)
            self.algo = self.converter.render(self.algo)
            self.algo = {
                'PPO': PPO,
                'SAC': SAC,
                'HER': HER
            }[self.algo.upper()]

        self.path = self.converter.render(
                value=self.path,
        )
        self.reward_scope = self.converter.render(self.reward_scope)
        if self.tensorboard_log is not None:
            self.tensorboard_log = self.converter.render(self.tensorboard_log)

        self.save_interval = self.converter.int(self.save_interval)
        self.log_interval = self.converter.int(self.log_interval)

        args = { }
        for key in ['batch_size', 'n_steps', 'learning_starts', 'verbose']:
            if key in self.attrs:
                args[key] = self.converter.int(self.attrs[key])
        
        if isinstance(self.algo, PPO.__class__):
            self.rl_setup_model(
                    model_class=PPO,
                    train=self.env.options.train,
                    path=self.path,
                    tensorboard_log=self.tensorboard_log,
                    tb_log_name=self.agent_name,
                    **args
            )
        elif isinstance(self.algo, SAC.__class__):
            self.rl_setup_model(
                    model_class=SAC,
                    train=self.env.options.train,
                    path=self.path,
                    tensorboard_log=self.tensorboard_log,
                    tb_log_name=self.agent_name,
                    **args
            )
        else:
            raise Exception('Unsupported algo type {}'.format(type(self.algo)))

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
        if self.reward_scope != '':
            return RLBaseNode.rl_gen_reward(self)
        return self.env.gen_reward(color=self.agent.color, previous=self.rl_info)

    def rl_reward_scope(self) -> str:
        return self.reward_scope

    def rl_gen_done(self) -> bool:
        info = self.env.gen_info()
        return info['terminated'] or info['truncated'] or not self.env.isopen

    def reset(self):
        if self.env.options.train:
            self.rl_model.logger.record("reward/round", self.rl_reward)
            self.rl_model.logger.record("wins/round", self.env.game_info[f'{self.agent.color}_wins'])
            self.rl_model.logger.record("loses/round", self.env.game_info[f'{self.agent.enemy_color}_wins'])
            self.rl_model.logger.record("draws/round", self.env.game_info['draws'])
            self.rl_model.logger.record("destroyed_count/round", self.agent.destroyed_count)
            self.rl_model.logger.record("missile_hit_enemy_count/round", self.agent.missile_hit_enemy_count)
            self.rl_model.logger.record("missile_miss_count/round", self.agent.missile_miss_count)
            self.rl_model.logger.record("missile_evade_success_count/round", self.agent.missile_evade_success_count)
            self.rl_model.logger.record("collided_aircraft_count/round", self.agent.aircraft_collided_count)

        super().reset()
        RLBaseNode.reset(self)

    def take_action(self):
        return self.rl_take_action(
                train=self.env.options.train,
                log_interval=self.log_interval,
                save_interval=self.save_interval,
                save_path=self.path
        )


class RLSwitcher(RLNode, Composite):
    """
    选择其中一个子节点来执行
    """

    def __init__(self, **kwargs):
        Composite.__init__(self, **kwargs)
        super().__init__(**kwargs)

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
            index = self.take_action()
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


class ReactiveRLSwitcher(RLSwitcher):
    """
    每次都会重新开始
    """

    def tick(self) -> typing.Iterator[Behaviour]:
        yield from self.switcher_tick(tick_again_status=[])


class RLSelector(RLNode, Selector):
    """
    将某个子节点作为开头，重置children并执行（事后回还原）
    self.children = self.children[index:] + self.children[:index]
    """

    def __init__(self, **kwargs):
        Selector.__init__(self, **kwargs)
        RLNode.__init__(**kwargs)
        self.init_children: list[py_trees.behaviour.Behaviour] = self.children[:]

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.children))

    def setup(self, **kwargs: typing.Any) -> None:
        self.init_children = self.children[:]
        super().setup(**kwargs)

    def tick(self) -> typing.Iterator[Behaviour]:
        index = self.take_action()
        self.children = self.init_children[index:] + self.init_children[:index]
        yield from Selector.tick(self)


class RLCondition(RLNode, pybts.Condition):
    """
    条件节点：用RL来判断是否需要执行
    """

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(2)

    def update(self) -> Status:
        action = self.take_action()
        if action == 0:
            return Status.FAILURE
        else:
            return Status.SUCCESS


class RLIntValue(RLNode, pybts.Condition):
    """
    强化学习int值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: int | str, low: int | str = 0, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.key = self.converter.render(self.key)
        self.high = self.converter.int(self.high)
        self.low = self.converter.int(self.low)

        assert self.high > self.low, "RLIntValue high must > low"

    def to_data(self):
        return {
            **super().to_data(),
            "key" : self.key,
            "high": self.high,
            "low" : self.low,
        }

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.high - self.low + 1, start=self.low)

    def update(self) -> Status:
        action = self.take_action()
        self.context[self.key] = int(action)
        return Status.SUCCESS


class RLFloatValue(RLNode, pybts.Condition):
    """
    强化学习float值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: float | str, low: float | str = 0, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.key = self.converter.render(self.key)
        self.high = self.converter.float(self.high)
        self.low = self.converter.float(self.low)

        assert self.high > self.low, "RLIntValue high must > low"

    def to_data(self):
        return {
            **super().to_data(),
            "key" : self.key,
            "high": self.high,
            "low" : self.low,
        }

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=self.low, high=self.high, shape=(1,))

    def update(self) -> Status:
        action = self.take_action()
        self.context[self.key] = float(action[0])
        return Status.SUCCESS


class RLAction(RLNode):
    def __init__(
            self,
            allow_actions: str = 'keep,go_to_location,fire_missile,go_home',
            **kwargs
    ):
        super().__init__(**kwargs)
        if allow_actions == '':
            allow_actions = 'keep,go_to_location,fire_missile,go_home'
        self.allow_actions = allow_actions.split(',')
        self.allow_actions = list(map(lambda x: Actions.build(x), self.allow_actions))

    def to_data(self):
        return {
            **super().to_data(),
            'allow_actions': self.allow_actions
        }

    def rl_action_space(self) -> gym.spaces.Space:
        # （action_type, distance, angle）
        return gym.spaces.Box(
                low=-1,
                high=1,
                shape=(3,)
        )

    def update(self) -> Status:
        action = self.rl_take_action(
                train=self.env.options.train,
                log_interval=self.log_interval,
                save_interval=self.save_interval,
                save_path=self.path
        )

        new_wpt = self.agent.waypoint.move(
                d=action[1] * self.agent.radar_radius,
                angle=action[2] * 180)

        action_type = Actions.extract_action_in_value_range(
                actions=self.allow_actions, value=action[0],
                value_range=(-1, 1))
        self.actions.put_nowait((action_type, new_wpt.x, new_wpt.y))
        return Status.SUCCESS


class RLFireAndGoToLocation(RLNode):
    def rl_action_space(self) -> gym.spaces.Space:
        # （action_type, distance, angle）
        return gym.spaces.Box(
                low=-1,
                high=1,
                shape=(3,)
        )

    def update(self) -> Status:
        action = self.take_action()

        new_wpt = self.agent.waypoint.move(
                d=action[1] * self.agent.radar_radius,
                angle=action[2] * 180)

        self.actions.put_nowait((Actions.go_to_location, new_wpt.x, new_wpt.y))

        if action[0] > 0:
            # 发射导弹，默认朝着最近的敌机发射
            self.actions.put_nowait((Actions.fire_missile, new_wpt.x, new_wpt.y))

        return Status.SUCCESS


class RLGoToLocation(RLNode):
    def rl_action_space(self) -> gym.spaces.Space:
        # （action_type, distance, angle）
        return gym.spaces.Box(
                low=-1,
                high=1,
                shape=(2,)
        )

    def update(self) -> Status:
        action = self.take_action()

        new_wpt = self.agent.waypoint.move(
                d=action[0] * self.agent.radar_radius,
                angle=action[1] * 180)

        self.actions.put_nowait((Actions.go_to_location, new_wpt.x, new_wpt.y))
        return Status.SUCCESS


class RLActionPPA(RLAction):
    """
    执行反向链式动作，子节点只能是PreCondition和PostCondition

    - PostCondition 后置条件：
        奖励为success_ratio
    - PreCondition 前置条件：
        惩罚为success_ratio-1

    【后置条件优先级大于前置条件】如果后置条件的success_ratio > 0，则不考虑前置条件

    状态：
    - if 后置条件存在:
        success_ratio == 1: SUCCESS
    - elif 前置条件存在且success_ratio > 0:
        返回RUNNING
    - else:
        返回FAILURE
    """

    def __init__(self, **kwargs):
        RLAction.__init__(self, **kwargs)
        self.pre_condition: pybts.PostCondition | None = None
        self.post_condition: pybts.PreCondition | None = None

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        assert len(self.children) in [1, 2], f'RLActionPPA: children count ({len(self.children)}) must be 1 or 2'
        for child in self.children:
            if isinstance(child, pybts.PostCondition):
                self.post_condition = child
            elif isinstance(child, pybts.PreCondition):
                self.pre_condition = child

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(3,))

    def rl_gen_reward(self) -> float:
        if self.post_condition is not None:
            self.post_condition.tick_once()
            if self.post_condition.success_ratio > 0:
                return self.post_condition.success_ratio

        if self.pre_condition is not None:
            self.pre_condition.tick_once()
            return self.pre_condition.success_ratio - 1

        return 0

    def update(self) -> Status:
        RLAction.update(self)

        if self.post_condition is not None:
            self.post_condition.tick_once()
            if self.post_condition.success_ratio == 1:
                return Status.SUCCESS
        if self.pre_condition is not None:
            self.pre_condition.tick_once()
            if self.pre_condition.success_ratio > 0:
                return Status.RUNNING
        return Status.FAILURE
