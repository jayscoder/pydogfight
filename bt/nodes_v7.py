from __future__ import annotations
from bt.base import *

"""
V7
- 共享了特征提取器
"""


class V7Init(BTPolicyNode):
    """
    V7初始化
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

        if self.agent.color == 'red':
            theta_range = [135, 225]
        else:
            theta_range = [-45, 45]
        theta = random.uniform(*theta_range)
        x = self.agent.radar_radius / 2 * math.cos(theta)
        y = self.agent.radar_radius / 2 * math.sin(theta)
        psi = random.random() * 360
        self.agent.waypoint = Waypoint.build(x=x, y=y, psi=psi)

        return Status.SUCCESS


class V7BaseSACNode:
    def rl_model_args(self: RLNode) -> dict:
        attrs = {
            'gamma'        : 0.995,
            'policy_kwargs': {
                'features_extractor_class' : V7FeatureExtractor1V1,
                'features_extractor_kwargs': {
                    'features_dim': 128,
                    'shared'      : V7FeatureExtractorShared.create(
                            name=self.agent.color,
                            features_dim=128,
                            shared=self.converter.bool(self.attrs.get('shared', False)))
                }
            }
        }

        attrs.update({
            'learning_starts': 120,
            'batch_size'     : 32,
            'train_freq'     : (1, "step"),
            'gradient_steps' : 1
        })

        return attrs


class V7SACGoToLocation1V1(RLGoToLocation, V7BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V7BaseSACNode.rl_model_args(self)


class V7SACFireAndGoToLocation1V1(RLFireAndGoToLocation, V7BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V7BaseSACNode.rl_model_args(self)


class V7SACCondition1V1(RLCondition, V7BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V7BaseSACNode.rl_model_args(self)


class V7FeatureExtractorShared(nn.Module):
    shared_dict = { }

    @classmethod
    def create(cls, name: str, features_dim: int, shared: bool) -> V7FeatureExtractorShared:
        if not shared:
            return V7FeatureExtractorShared(features_dim=features_dim)
        key = f'{name}_{features_dim}'
        if key in cls.shared_dict:
            return cls.shared_dict[key]
        else:
            return V7FeatureExtractorShared(features_dim=features_dim)

    def __init__(self, features_dim: int):
        super().__init__()
        # 添加一个线性层用来处理提取的特征
        self.fc = nn.Linear(1 + 1 + 3 + 3 * ObsUtils.WATCH_MISSILES, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
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


class V7FeatureExtractor1V1(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128, shared: nn.Module = None):
        super().__init__(observation_space, features_dim)
        # 共享特征提取器
        self.shared = shared

    def forward(self, observations: torch.Tensor):
        return self.shared(observations)

# if __name__ == '__main__':
#     spac = gym.spaces.Box(low=0, high=2, shape=(1,))
#     print(bool(spac.sample() > 10))
