from __future__ import annotations

from py_trees.behaviour import Behaviour
from py_trees.common import Status
from pybts.rl.nodes import Reward
from stable_baselines3 import PPO, SAC, HerReplayBuffer, DQN, DDPG, TD3
import py_trees
import gymnasium as gym
from pybts.rl import RLBaseNode
from pybts.rl.common import is_off_policy_algo, is_on_policy_algo
import typing
from pybts.rl.logger import TensorboardLogger
from pybts.composites import *
from pydogfight.core.actions import Actions
from pydogfight.policy.bt.base_class import *
from custom.thesis_ppo import *


class RLNode(BTPolicyNode, RLBaseNode, ABC):
    """

    deterministic:
        true: 确定性动作意味着对于给定的状态或观测，策略总是返回相同的动作。没有随机性或变化性涉及，每次给定相同的输入状态，输出（即动作）总是一样的。
            在实际应用中，确定性选择通常用于部署阶段，当你希望模型表现出最稳定、可预测的行为时，例如在测试或实际运行环境中。
        false: 随机性动作则意味着策略在给定的状态下可能产生多种可能的动作。这通常是通过策略输出的概率分布实现的，例如，一个使用softmax输出层的神经网络可能会对每个可能的动作分配一个概率，然后根据这个分布随机选择动作。
            随机性在训练阶段特别有用，因为它可以增加探索，即允许代理（agent）尝试和学习那些未必立即最优但可能长期更有益的动作。这有助于策略避免陷入局部最优并更全面地学习环境。
    """

    def __init__(
            self,
            path: str = '',
            algo: str = 'PPO',
            domain: str = '',
            save_path: str = '',  # 空代表不保存
            save_interval: int | str = 0,
            deterministic: bool | str = False,
            train: bool | str = False,
            tensorboard_log: str = '',
            **kwargs
    ):
        super().__init__(**kwargs)
        RLBaseNode.__init__(self)
        self.algo = algo
        self.domain = domain  # 如果domain设置成default或其他不为空的值，则认为奖励要从context.rl_reward[domain]中拿
        self.path = path
        self.save_path = save_path
        self.save_interval = save_interval
        self.deterministic = deterministic
        self.train = train
        self.tensorboard_log = tensorboard_log

    def to_data(self):
        return {
            **super().to_data(),
            **RLBaseNode.to_data(self),
            'algo'         : str(self.algo),
            'path'         : self.path,
            'domain'       : self.domain,
            'save_interval': self.save_interval,
        }

    def rl_model_args(self) -> dict:
        return { }

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)

        self.path = self.converter.render(
                value=self.path,
        )
        print(f'RLNode({self.name}).path=', self.path)
        self.domain = self.converter.render(self.domain)
        self.save_interval = self.converter.int(self.save_interval)
        self.algo = self.converter.render(self.algo).upper()
        if self.tensorboard_log != '':
            self.tensorboard_log = self.converter.render(self.tensorboard_log)

        args = self.rl_model_args()
        for key in ['batch_size', 'n_steps', 'learning_starts', 'verbose']:
            if key in self.attrs:
                args[key] = self.converter.int(self.attrs[key])

        self.setup_model(algo=self.algo, **args)

    def setup_model(self, algo: str, **kwargs):
        policy = kwargs.get('policy', 'MlpPolicy')

        tensorboard_logger = TensorboardLogger(folder=self.tensorboard_log, verbose=0)

        if algo == 'THESIS-PPO':
            if 'PPOSharedValueModel' in self.context:
                self.rl_setup_model(
                        policy=policy,
                        model_class=ThesisPPOOnlyAction,
                        train=True,
                        path=self.path,
                        logger=tensorboard_logger,
                        **kwargs
                )
                # 价值网络使用context里的
                assert isinstance(self.rl_model.policy, ThesisActorCriticPolicy)
                ppo_shared_value_model = self.context['PPOSharedValueModel']
                assert isinstance(ppo_shared_value_model, ThesisPPOOnlyValue)
                assert isinstance(ppo_shared_value_model.policy, ThesisActorCriticPolicy)
                self.rl_model.policy.features_extractor = ppo_shared_value_model.policy.features_extractor
                self.rl_model.policy.mlp_extractor.value_net = ppo_shared_value_model.policy.mlp_extractor.value_net
                self.rl_model.policy.value_net = ppo_shared_value_model.policy.value_net
            else:
                self.rl_setup_model(
                        policy=policy,
                        model_class=ThesisPPO,
                        train=True,
                        path=self.path,
                        logger=tensorboard_logger,
                        **kwargs
                )
        elif algo == 'THESIS-PPO-VALUE':
            self.rl_setup_model(
                    policy=policy,
                    model_class=ThesisPPOOnlyValue,
                    train=True,
                    path=self.path,
                    logger=tensorboard_logger,
                    **kwargs
            )
            self.context['PPOSharedValueModel'] = self.rl_model  # 共享价值网络
        elif algo == 'PPO':
            self.rl_setup_model(
                    policy=policy,
                    model_class=PPO,
                    train=True,
                    path=self.path,
                    logger=tensorboard_logger,
                    **kwargs
            )
        elif algo == 'SAC':
            self.rl_setup_model(
                    policy=policy,
                    model_class=SAC,
                    train=True,  # 在训练过程中可能会开/闭某个节点的训练，所以一开始初始化都默认开启训练
                    path=self.path,
                    logger=tensorboard_logger,
                    **kwargs
            )
        elif algo == 'SAC-HER':
            self.rl_setup_model(
                    policy=policy,
                    model_class=SAC,
                    train=True,
                    path=self.path,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy='future',
                    ),
                    logger=tensorboard_logger,
                    **kwargs
            )
        elif algo == 'TD3':
            self.rl_setup_model(
                    policy=policy,
                    model_class=TD3,
                    train=True,
                    path=self.path,
                    **kwargs
            )
        elif algo == 'TD3-HER':
            self.rl_setup_model(
                    policy=policy,
                    model_class=TD3,
                    train=True,
                    path=self.path,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy='future',
                    ),
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
        if self.domain != '':
            return RLBaseNode.rl_gen_reward(self)
        return self.env.gen_reward(color=self.agent.color, previous=self.rl_info)

    def rl_domain(self) -> str:
        return self.domain

    def rl_gen_done(self) -> bool:
        info = self.env.gen_info()
        return info['terminated'] or info['truncated'] or not self.env.isopen

    def rl_device(self) -> str:
        return self.env.options.device

    def reset(self):
        # self.rl_model.logger.record("final_reward", self.rl_reward)
        # self.rl_model.logger.record("return", self.rl_accum_reward)

        # self.rl_model.logger.record("win", self.env.game_info[self.agent.color]['win'])
        # self.rl_model.logger.record("lose", self.env.game_info[self.agent.color]['lose'])
        # self.rl_model.logger.record("draws", self.env.game_info['draws'])
        # self.rl_model.logger.record("destroyed_count", self.agent.destroyed_count)
        # self.rl_model.logger.record("missile_hit_enemy_count", self.agent.missile_hit_enemy_count)
        # self.rl_model.logger.record("missile_miss_count", self.agent.missile_miss_count),
        # self.rl_model.logger.record("missile_hit_self_count", self.agent.missile_hit_self_count),
        # self.rl_model.logger.record("missile_fired_count", self.agent.missile_fired_count),
        # self.rl_model.logger.record("missile_evade_success_count", self.agent.missile_evade_success_count)
        # self.rl_model.logger.record("aircraft_collided_count", self.agent.aircraft_collided_count)

        if self.env.episode > 0 and self.save_interval > 0 and self.env.episode % self.save_interval == 0 and self.save_path != '':
            save_path = self.converter.render(self.save_path)
            self.rl_model.save(path=save_path)

        super().reset()
        RLBaseNode.reset(self)

    def take_action(self):
        return self.rl_take_action(
                train=self.converter.bool(self.train),
                deterministic=self.converter.bool(self.deterministic),
        )

    def save_model(self, filepath: str = ''):
        if filepath == '':
            filepath = self.converter.render(self.save_path)
        else:
            filepath = self.converter.render(filepath)
        self.rl_model.save(path=filepath)


class RLComposite(RLNode, Composite):
    def __init__(self, **kwargs):
        Composite.__init__(self, **kwargs)
        RLNode.__init__(self, **kwargs)

    def rl_action_space(self) -> gym.spaces.Space:
        if is_off_policy_algo(self.algo):
            return gym.spaces.Box(low=0, high=len(self.children))
        return gym.spaces.Discrete(len(self.children))

    def gen_index(self) -> int:
        if is_off_policy_algo(self.algo):
            index = int(self.take_action()[0]) % len(self.children)
        else:
            index = self.take_action()
        self.put_update_message(f'gen_index index={index} train={self.train}')
        return index


class RLSwitcher(RLComposite, Switcher):
    """
    选择其中一个子节点来执行
    """

    @property
    def allow_action_mask(self):
        # 是否允许动作掩码
        return self.converter.bool(self.attrs.get('allow_action_mask', True))

    def tick(self) -> typing.Iterator[Behaviour]:
        # if self.exp_fill and self.train and self.status in self.tick_again_status():
        #     yield from self.switch_tick(index=self.gen_index(), tick_again_status=self.tick_again_status())
        #     self.rl_action = self.current_index  # 保存动作
        # else:
        return self.switch_tick(index=self.gen_index(), tick_again_status=self.tick_again_status())

    def rl_gen_action_mask(self):
        if not self.allow_action_mask:
            return None
        action_mask = np.ones((len(self.children, )))
        for i, child in enumerate(self.children):
            if isinstance(child, pybts.Sequence):
                status = child.peek_tick()
                if status == Status.FAILURE:
                    action_mask[i] = 0

        # 如果当前子节点的状态是0且
        tick_again_status = self.tick_again_status()

        if self.status in tick_again_status:
            index = self.children.index(self.current_child)

            if action_mask[index] == 1:
                # 该子动作没有被屏蔽，则将剩余动作全部屏蔽
                action_mask.fill(0)  # 先将所有动作置为屏蔽
                action_mask[index] = 1  # 仅当前子节点可执行

        return action_mask


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
        if is_off_policy_algo(self.algo):
            return gym.spaces.Box(low=0, high=2, shape=(1,))
        return gym.spaces.Discrete(2)

    def update(self) -> Status:
        if is_off_policy_algo(self.algo):
            action = self.take_action()[0]
        else:
            action = self.take_action()
        if action >= 1:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class RLIntValue(RLNode):
    """
    强化学习int值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: int | str, low: int | str = 0, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        self.key = self.converter.render(self.key)
        self.high = self.converter.int(self.high)
        self.low = self.converter.int(self.low)

        super().setup(**kwargs)

        assert self.high > self.low, "RLIntValue high must > low"

    def to_data(self):
        return {
            **super().to_data(),
            "key" : self.key,
            "high": self.high,
            "low" : self.low,
        }

    def rl_action_space(self) -> gym.spaces.Space:
        self.high = self.converter.int(self.high)
        self.low = self.converter.int(self.low)
        return gym.spaces.Discrete(self.high - self.low + 1)

    def update(self) -> Status:
        action = self.take_action()
        self.context[self.key] = int(action) + self.low
        return Status.SUCCESS


class RLFloatValue(RLNode):
    """
    强化学习float值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: float | str, low: float | str = 0, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        self.key = self.converter.render(self.key)
        self.high = self.converter.float(self.high)
        self.low = self.converter.float(self.low)

        super().setup(**kwargs)

        assert self.high > self.low, "RLFloatValue high must > low"

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
        action = self.take_action()[0]
        self.context[self.key] = float(action)
        return Status.SUCCESS


class RLFloatArrayValue(RLNode):
    """
    强化学习float值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: float | str, low: float | str = 0, length: int | str = 1, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low
        self.length = int(length)

    def setup(self, **kwargs: typing.Any) -> None:
        self.key = self.converter.render(self.key)
        self.high = self.converter.float(self.high)
        self.low = self.converter.float(self.low)
        self.length = self.converter.int(self.length)
        super().setup(**kwargs)
        assert self.high > self.low, "RLFloatValue high must > low"

    def to_data(self):
        return {
            **super().to_data(),
            "key"   : self.key,
            "high"  : self.high,
            "low"   : self.low,
            'length': self.length
        }

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=self.low, high=self.high, shape=(self.length,))

    def update(self) -> Status:
        action = self.take_action()
        self.context[self.key] = action.tolist()
        return Status.SUCCESS


#
# class RLIntArrayValue(RLNode):
#     """
#     强化学习int值生成，将生成的值保存到context[key]里
#     """
#
#     def __init__(self, key: str, high: int | str, low: int | str = 0, length: int | str = 1, **kwargs):
#         super().__init__(**kwargs)
#         self.key = key
#         self.high = high
#         self.low = low
#         self.length = int(length)
#
#     def setup(self, **kwargs: typing.Any) -> None:
#         self.key = self.converter.render(self.key)
#         self.high = self.converter.int(self.high)
#         self.low = self.converter.int(self.low)
#         self.length = self.converter.int(self.length)
#         super().setup(**kwargs)
#         assert self.high > self.low, "RLIntArrayValue high must > low"
#
#     def to_data(self):
#         return {
#             **super().to_data(),
#             "key"   : self.key,
#             "high"  : self.high,
#             "low"   : self.low,
#             'length': self.length
#         }
#
#     def rl_action_space(self) -> gym.spaces.Space:
#         return gym.spaces.MultiDiscrete()
#
#     def update(self) -> Status:
#         action = self.take_action()
#         self.context[self.key] = action.tolist()
#         return Status.SUCCESS
#

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
                shape=(2,)
        )

    def update(self) -> Status:
        action = self.take_action()

        new_wpt = self.agent.waypoint.move(
                d=self.agent.radar_radius,
                angle=action[1] * 180)

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
                shape=(1,)
        )

    def update(self) -> Status:
        action = self.take_action()

        new_wpt = self.agent.waypoint.move(
                d=self.agent.radar_radius,
                angle=action[0] * 180)

        self.actions.put_nowait((Actions.go_to_location, new_wpt.x, new_wpt.y))
        return Status.SUCCESS


class RLReward(RLNode, Reward):
    def __init__(self, high: int | str, low: int | str = 0, **kwargs):
        Reward.__init__(self, **kwargs)
        RLNode.__init__(self, **kwargs)
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.high = self.converter.float(self.high)
        self.low = self.converter.float(self.low)

    def to_data(self):
        return {
            **super().to_data(),
            **Reward.to_data(self),
            'low' : self.low,
            'high': self.high
        }

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=self.low, high=self.high, shape=(1,))

    def update(self) -> Status:
        self.reward = self.take_action()
        print('RLReward', self.reward)
        return Reward.update(self)


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
