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
from stable_baselines3 import PPO, SAC, HerReplayBuffer, DQN, DDPG, TD3
import py_trees
import gymnasium as gym
from pybts.rl import RLBaseNode
import typing
import jinja2
from pydogfight.core.actions import Actions


class RLNode(BTPolicyNode, RLBaseNode, ABC):
    """

    deterministic:
        true: 确定性动作意味着对于给定的状态或观测，策略总是返回相同的动作。没有随机性或变化性涉及，每次给定相同的输入状态，输出（即动作）总是一样的。
            在实际应用中，确定性选择通常用于部署阶段，当你希望模型表现出最稳定、可预测的行为时，例如在测试或实际运行环境中。
        false: 随机性动作则意味着策略在给定的状态下可能产生多种可能的动作。这通常是通过策略输出的概率分布实现的，例如，一个使用softmax输出层的神经网络可能会对每个可能的动作分配一个概率，然后根据这个分布随机选择动作。
            随机性在训练阶段特别有用，因为它可以增加探索，即允许代理（agent）尝试和学习那些未必立即最优但可能长期更有益的动作。这有助于策略避免陷入局部最优并更全面地学习环境。
    """

    def __init__(self,
                 path: str = '',
                 algo: str = 'PPO',
                 reward_scope: str = '',
                 tensorboard_log: typing.Optional[str] = None,
                 log_interval: int | str = 10,
                 save_path: str = '',  # 空代表在path那里保存
                 save_interval: int | str = 10,
                 deterministic: bool | str = False,
                 train: bool | str = False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        RLBaseNode.__init__(self)
        self.algo = algo
        self.reward_scope = reward_scope  # 如果scope设置成default或其他不为空的值，则认为奖励要从context.rl_reward[scope]中拿
        self.path = path
        self.tensorboard_log = tensorboard_log
        self.log_interval = log_interval
        self.save_path = save_path
        self.save_interval = save_interval
        self.deterministic = deterministic
        self.train = train

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

    def rl_model_args(self) -> dict:
        return { }

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)

        self.path = self.converter.render(
                value=self.path,
        )
        self.reward_scope = self.converter.render(self.reward_scope)
        if self.tensorboard_log is not None:
            self.tensorboard_log = self.converter.render(self.tensorboard_log)

        self.save_interval = self.converter.int(self.save_interval)
        self.log_interval = self.converter.int(self.log_interval)

        self.algo = self.converter.render(self.algo).upper()

        args = self.rl_model_args()
        for key in ['batch_size', 'n_steps', 'learning_starts', 'verbose']:
            if key in self.attrs:
                args[key] = self.converter.int(self.attrs[key])

        self.setup_model(algo=self.algo, **args)

    def setup_model(self, algo: str, **kwargs):
        if algo == 'PPO':
            self.rl_setup_model(
                    model_class=PPO,
                    train=True,
                    path=self.path,
                    tensorboard_log=self.tensorboard_log,
                    tb_log_name=self.agent_name,
                    **kwargs
            )
        elif algo == 'SAC':
            self.rl_setup_model(
                    model_class=SAC,
                    train=True,  # 在训练过程中可能会开/闭某个节点的训练，所以一开始初始化都默认开启训练
                    path=self.path,
                    tensorboard_log=self.tensorboard_log,
                    tb_log_name=self.agent_name,
                    **kwargs
            )
        elif algo == 'SAC-HER':
            self.rl_setup_model(
                    model_class=SAC,
                    train=True,
                    path=self.path,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy='future',
                    ),
                    tensorboard_log=self.tensorboard_log,
                    tb_log_name=self.agent_name,
                    **kwargs
            )
        elif algo == 'TD3':
            self.rl_setup_model(
                    model_class=TD3,
                    train=True,
                    path=self.path,
                    tensorboard_log=self.tensorboard_log,
                    tb_log_name=self.agent_name,
                    **kwargs
            )
        elif algo == 'TD3-HER':
            self.rl_setup_model(
                    model_class=TD3,
                    train=True,
                    path=self.path,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy='future',
                    ),
                    tensorboard_log=self.tensorboard_log,
                    tb_log_name=self.agent_name,
                    **kwargs
            )
        else:
            raise Exception(f'Unsupported algo type {algo}')

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

    def rl_device(self) -> str:
        return self.env.options.device

    def reset(self):
        self.rl_model.logger.record("episode", self.env.episode)
        self.rl_model.logger.record("returns", self.rl_accum_reward)
        self.rl_model.logger.record("wins", self.env.game_info[f'{self.agent.color}_wins'])
        self.rl_model.logger.record("loses", self.env.game_info[f'{self.agent.enemy_color}_wins'])
        self.rl_model.logger.record("draws", self.env.game_info['draws'])
        self.rl_model.logger.record("destroyed_count", self.agent.destroyed_count)
        self.rl_model.logger.record("missile_hit_enemy_count", self.agent.missile_hit_enemy_count)
        self.rl_model.logger.record("missile_miss_count", self.agent.missile_miss_count),
        self.rl_model.logger.record("missile_hit_self_count", self.agent.missile_hit_self_count),
        self.rl_model.logger.record("missile_fired_count", self.agent.missile_fired_count),
        self.rl_model.logger.record("missile_evade_success_count", self.agent.missile_evade_success_count)
        self.rl_model.logger.record("aircraft_collided_count", self.agent.aircraft_collided_count)

        if self.env.episode > 0 and self.env.episode % self.converter.int(self.save_interval) == 0:
            save_path = self.converter.render(self.save_path)
            if save_path == '':
                save_path = self.path
            self.rl_model.save(path=save_path)
        
        super().reset()
        RLBaseNode.reset(self)

    def take_action(self):
        return self.rl_take_action(
                train=self.converter.bool(self.train),
                log_interval=self.log_interval,
                deterministic=self.converter.bool(self.deterministic)
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
        action = self.take_action()

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
