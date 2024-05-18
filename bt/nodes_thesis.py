from __future__ import annotations

import gym.spaces
import pybts

from bt.base import *
from pydogfight.policy.bt.nodes_conditions import *
from custom.thesis_ppo import ThesisRolloutBuffer

"""
Thesis
- 测试强化学习优化参数
"""


class ThesisInit(BTPolicyNode):
    """
    论文环境初始化
    随机初始化自己的位置在战场中心雷达半径附近，方便双方一开始就能相遇在双方的雷达半径内
    Randomly initializes the agent's position near the center of the battlefield radar radius,
    facilitating an early encounter within each other's radar range.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inited = False

    def reset(self):
        super().reset()
        self.inited = False

    def update(self):
        if self.inited:
            return Status.SUCCESS
        self.inited = True

        # if self.agent.color == 'red':
        theta_range = [0, 360]
        # else:
        #     theta_range = [-45, 45]
        theta = math.radians(random.uniform(*theta_range))
        x = self.agent.radar_radius / 2 * math.cos(theta)
        y = self.agent.radar_radius / 2 * math.sin(theta)
        psi = random.random() * 360
        self.agent.waypoint = Waypoint.build(x=x, y=y, psi=psi)
        return Status.SUCCESS


class ThesisBasePPONode:
    def rl_model_args(self: RLNode) -> dict:
        # batch_size = self.converter.int(self.attrs.get('batch_size', 128))
        # attrs = {
        #     'gamma': 0.99,
        # }
        # attrs.update({
        #     'batch_size': batch_size,
        # })
        return {
            'n_steps'             : self.converter.int(self.attrs.get('n_steps', 2048)),
            'rollout_buffer_class': ThesisRolloutBuffer
        }

    def wrap_obs(self, obs):
        # self_can_fire_missile = obs[0:1, 5]  # 自己能不能发射导弹
        # enemy_can_fire_missile = obs[1:2, 5]  # 敌人能不能发射导弹
        # enemy_rel_polar_waypoint = obs[1:2, 2:5]  # 敌人相对于自己的极坐标
        # missile_rel_polar_waypoint = obs[2:, 2:5]  # 敌人发射的导弹的极坐标
        #
        # # 使用 numpy 将这些特征合并成一个数组
        # combined_features = np.concatenate([
        #     self_can_fire_missile,
        #     enemy_can_fire_missile,
        #     enemy_rel_polar_waypoint.reshape(-1),  # 展平极坐标特征
        #     missile_rel_polar_waypoint.reshape(-1)  # 展平导弹的极坐标特征
        # ])

        return obs

    @classmethod
    def wrap_observation_space(cls, obs_space) -> gym.spaces.Space:
        return obs_space
        # return gym.spaces.Box(low=-10, high=10, shape=(20,), dtype=np.float32)


class ThesisPPOSwitcher(RLSwitcher, ThesisBasePPONode):
    def __init__(self, **kwargs):
        ThesisBasePPONode.__init__(self)
        super().__init__(algo='THESIS-PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return ThesisBasePPONode.rl_model_args(self)

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.children))

    def rl_gen_obs(self):
        return self.wrap_obs(super().rl_gen_obs())

    def rl_observation_space(self) -> gym.spaces.Space:
        return ThesisBasePPONode.wrap_observation_space(super().rl_observation_space())


class ThesisPPOValue(RLNode, ThesisBasePPONode):
    def __init__(self, **kwargs):
        ThesisBasePPONode.__init__(self)
        super().__init__(algo='THESIS-PPO-VALUE', **kwargs)

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        print('ThesisPPOValue.path', self.converter.render(self.path), self.path)

    def rl_model_args(self) -> dict:
        return ThesisBasePPONode.rl_model_args(self)

    def rl_gen_obs(self):
        obs = super().rl_gen_obs()
        return self.wrap_obs(obs)

    def rl_observation_space(self) -> gym.spaces.Space:
        return ThesisBasePPONode.wrap_observation_space(super().rl_observation_space())

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(2)  # 随便选的一个动作空间

    def update(self) -> Status:
        self.take_action()
        return Status.SUCCESS


class ThesisPPOFloatArray(RLFloatArrayValue, ThesisBasePPONode):
    def __init__(self, **kwargs):
        ThesisBasePPONode.__init__(self)
        super().__init__(algo='THESIS-PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return ThesisBasePPONode.rl_model_args(self)

    def rl_gen_obs(self):
        obs = super().rl_gen_obs()
        return self.wrap_obs(obs)

    def rl_observation_space(self) -> gym.spaces.Space:
        return ThesisBasePPONode.wrap_observation_space(super().rl_observation_space())


class ThesisPPOIntValue(RLIntValue, ThesisBasePPONode):
    def __init__(self, **kwargs):
        ThesisBasePPONode.__init__(self)
        super().__init__(algo='THESIS-PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return ThesisBasePPONode.rl_model_args(self)

    def rl_gen_obs(self):
        obs = super().rl_gen_obs()
        return self.wrap_obs(obs)

    def rl_observation_space(self) -> gym.spaces.Space:
        return ThesisBasePPONode.wrap_observation_space(super().rl_observation_space())


class ThesisPPOCondition(RLCondition, ThesisBasePPONode):
    def __init__(self, **kwargs):
        ThesisBasePPONode.__init__(self)
        super().__init__(algo='THESIS-PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return ThesisBasePPONode.rl_model_args(self)

    def rl_gen_obs(self):
        obs = super().rl_gen_obs()
        return self.wrap_obs(obs)

    def rl_observation_space(self) -> gym.spaces.Space:
        return ThesisBasePPONode.wrap_observation_space(super().rl_observation_space())


class ThesisPPOFireAndGoToLocation(RLFireAndGoToLocation, ThesisBasePPONode):
    def __init__(self, **kwargs):
        ThesisBasePPONode.__init__(self)
        super().__init__(algo='THESIS-PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return ThesisBasePPONode.rl_model_args(self)

    def rl_gen_obs(self):
        obs = super().rl_gen_obs()
        return self.wrap_obs(obs)

    def rl_observation_space(self) -> gym.spaces.Space:
        return ThesisBasePPONode.wrap_observation_space(super().rl_observation_space())


class ThesisPPOGoToLocation(RLGoToLocation, ThesisBasePPONode):
    def __init__(self, **kwargs):
        ThesisBasePPONode.__init__(self)
        super().__init__(algo='THESIS-PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return ThesisBasePPONode.rl_model_args(self)

    def rl_gen_obs(self):
        obs = super().rl_gen_obs()
        return self.wrap_obs(obs)

    def rl_observation_space(self) -> gym.spaces.Space:
        return ThesisBasePPONode.wrap_observation_space(super().rl_observation_space())
