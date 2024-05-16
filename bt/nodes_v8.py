from __future__ import annotations
from bt.base import *

"""
V8
- 测试强化学习优化参数
"""


class V8Init(BTPolicyNode):
    """
    V8初始化
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


class V8BaseSACNode:
    def rl_model_args(self: RLNode) -> dict:
        features_dim = self.converter.int(self.attrs.get('features_dim', 128))
        learning_starts = self.converter.int(self.attrs.get('learning_starts', 128))
        batch_size = self.converter.int(self.attrs.get('batch_size', 32))
        attrs = {
            'gamma'        : 0.995,
            'policy_kwargs': {
                'features_extractor_class' : V8FeatureExtractor1V1,
                'features_extractor_kwargs': {
                    'features_dim': features_dim,
                }
            }
        }

        attrs.update({
            'learning_starts': learning_starts,
            'batch_size'     : batch_size,
            'train_freq'     : (1, "step"),
            'gradient_steps' : 1
        })

        return attrs


class V8BasePPONode:
    def rl_model_args(self: RLNode) -> dict:
        features_dim = self.converter.int(self.attrs.get('features_dim', 128))
        batch_size = self.converter.int(self.attrs.get('batch_size', 64))
        attrs = {
            'gamma'        : 0.995,
            'policy_kwargs': {
                'features_extractor_class' : V8FeatureExtractor1V1,
                'features_extractor_kwargs': {
                    'features_dim': features_dim,
                }
            }
        }
        attrs.update({
            'batch_size': batch_size,
        })
        return attrs


class V8BaseDQNNode:
    def rl_model_args(self: RLNode) -> dict:
        features_dim = self.converter.int(self.attrs.get('features_dim', 128))
        batch_size = self.converter.int(self.attrs.get('batch_size', 64))
        attrs = {
            'gamma'        : 0.995,
            'policy_kwargs': {
                'features_extractor_class' : V8FeatureExtractor1V1,
                'features_extractor_kwargs': {
                    'features_dim': features_dim,
                }
            }
        }
        attrs.update({
            'batch_size': batch_size,
        })
        return attrs


class V8FeatureExtractor1V1(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 共享特征提取器
        self.fc = nn.Linear(1 + 1 + 3 + 3 * ObsUtils.WATCH_MISSILES, features_dim)

    def forward(self, obs: torch.Tensor):
        self_can_fire_missile = obs[:, 0:1, 5]  # 自己能不能发射导弹
        enemy_can_fire_missile = obs[:, 1:2, 5]  # 敌人能不能发射导弹
        enemy_rel_polar_waypoint = obs[:, 1:2, 2:5]  # 敌人相对于自己的极坐标
        missile_rel_polar_waypoint = obs[:, 2:, 2:5]  # 敌人发射的导弹的极坐标

        combined_features = torch.cat([
            self_can_fire_missile,  # 增加一个维度以匹配其他特征的形状
            enemy_can_fire_missile,
            enemy_rel_polar_waypoint.reshape(obs.shape[0], -1),
            missile_rel_polar_waypoint.reshape(obs.shape[0], -1)  # 展平导弹的极坐标特征
        ], dim=1)

        # 将特征向量通过一个全连接层进行进一步处理
        processed_features = self.fc(combined_features)
        return processed_features


class V8SACFloatArrayValue(RLFloatArrayValue):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BaseSACNode.rl_model_args(self)


class V8SACFloatValue(RLFloatValue):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BaseSACNode.rl_model_args(self)


class V8SACSwitcher(RLSwitcher, V8BaseSACNode):
    def __init__(self, **kwargs):
        V8BaseSACNode.__init__(self)
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BaseSACNode.rl_model_args(self)

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0, high=len(self.children))

    def gen_index(self) -> int:
        index = int(self.take_action()[0]) % len(self.children)
        return index


class V8PPOSwitcher(RLSwitcher, V8BasePPONode):
    def __init__(self, **kwargs):
        V8BasePPONode.__init__(self)
        super().__init__(algo='PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BasePPONode.rl_model_args(self)

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0, high=len(self.children))

    def gen_index(self) -> int:
        index = int(self.take_action()[0]) % len(self.children)
        return index


class V8PPOFloatArray(RLFloatArrayValue, V8BasePPONode):
    def __init__(self, **kwargs):
        V8BasePPONode.__init__(self)
        super().__init__(algo='PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BasePPONode.rl_model_args(self)


class V8SACFireAndGoToLocation1V1(RLFireAndGoToLocation, V8BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BaseSACNode.rl_model_args(self)


class V8PPOFireAndGoToLocation1V1(RLFireAndGoToLocation, V8BasePPONode):
    def __init__(self, **kwargs):
        super().__init__(algo='PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BasePPONode.rl_model_args(self)


class V8PPOGoToLocation1V1(RLGoToLocation, V8BasePPONode):
    def __init__(self, **kwargs):
        super().__init__(algo='PPO', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BasePPONode.rl_model_args(self)

    def take_action(self):
        # print('policy', self.rl_model.policy)
        return super().take_action()


class V8SACGoToLocation1V1(RLGoToLocation, V8BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V8BaseSACNode.rl_model_args(self)

    def take_action(self):
        # print('actor', self.rl_model.policy.actor)
        # print('critic', self.rl_model.policy.critic)
        return super().take_action()

# if __name__ == '__main__':
#     spac = gym.spaces.Box(low=0, high=2, shape=(1,))
#     print(bool(spac.sample() > 10))
