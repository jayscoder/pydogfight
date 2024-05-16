from __future__ import annotations

import math

from bt.base import *

"""
V7
- 测试共享特征提取器
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

        theta = math.radians(random.uniform(*theta_range))
        x = self.agent.radar_radius / 2 * math.cos(theta)
        y = self.agent.radar_radius / 2 * math.sin(theta)
        psi = random.random() * 360
        self.agent.waypoint = Waypoint.build(x=x, y=y, psi=psi)

        return Status.SUCCESS


class V7BaseSACNode:
    def rl_model_args(self: RLNode) -> dict:
        features_dim = self.converter.int(self.attrs.get('features_dim', 128))
        learning_starts = self.converter.int(self.attrs.get('learning_starts', 128))
        batch_size = self.converter.int(self.attrs.get('batch_size', 32))
        lstm = self.converter.bool(self.attrs.get('lstm', False))
        nof = self.converter.bool(self.attrs.get('nof', False))
        gamma = self.converter.float(self.attrs.get('gamma', 0.99))
        if nof:
            attrs = {
            }
        elif lstm:
            attrs = {
                'policy_kwargs': {
                    'features_extractor_class' : V7FeatureExtractorLSTM1V1,
                    'features_extractor_kwargs': {
                        'features_dim': features_dim,
                    }
                }
            }
        else:
            attrs = {
                'policy_kwargs': {
                    'features_extractor_class' : V7FeatureExtractor1V1,
                    'features_extractor_kwargs': {
                        'features_dim': features_dim,
                        'shared'      : V7FeatureExtractorShared.create(
                                name=self.agent.color,
                                features_dim=features_dim,
                                shared=self.converter.bool(self.attrs.get('shared', False)))
                    }
                }
            }

        attrs.update({
            'learning_starts': learning_starts,
            'batch_size'     : batch_size,
            'train_freq'     : (1, "step"),
            'gradient_steps' : 1,
            # 'gamma'          : gamma
        })

        return attrs

    def reset(self: RLNode):
        lstm = self.converter.bool(self.attrs.get('lstm', False))
        if lstm and self.rl_model is not None and self.rl_model.policy.features_extractor is not None:
            print('RLModelPolicy', self.rl_model.policy)
            self.rl_model.policy.features_extractor.reset_states()


class V7SACGoToLocation1V1(RLGoToLocation, V7BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V7BaseSACNode.rl_model_args(self)

    def reset(self):
        super().reset()
        V7BaseSACNode.reset(self)


class V7SACFireAndGoToLocation1V1(RLFireAndGoToLocation, V7BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V7BaseSACNode.rl_model_args(self)

    def reset(self):
        super().reset()
        V7BaseSACNode.reset(self)


class V7SACCondition1V1(RLCondition, V7BaseSACNode):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)

    def rl_model_args(self) -> dict:
        return V7BaseSACNode.rl_model_args(self)

    def reset(self):
        super().reset()
        V7BaseSACNode.reset(self)


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
        self.fc = nn.Linear(4 * 2 + 4 * ObsUtils.WATCH_MISSILES, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        aircrafts = obs[:, 0:2, 2:6]  # 自己和敌人
        missiles = obs[:, 2:, 2:6]  # 敌人发射的导弹

        combined_features = torch.cat([
            aircrafts.reshape(obs.shape[0], -1),
            missiles.reshape(obs.shape[0], -1)  # 展平导弹的极坐标特征
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


class V7FeatureExtractorLSTM1V1(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(4 * 2 + 4 * ObsUtils.WATCH_MISSILES, features_dim, 2, batch_first=True)
        self.hidden_cell = None
        # self.label = nn.Linear(64, features_dim)

    def forward(self, obs: torch.Tensor, hidden_state=None, cell_state=None):
        obs = obs[:, :, 2:6].reshape(obs.shape[0], 1, -1)  # 自己和敌人

        print('hidden_state', hidden_state)

        if hidden_state is not None and cell_state is not None:
            # If states are provided, use them
            output, (final_hidden_state, final_cell_state) = self.lstm(obs, (hidden_state, cell_state))
        else:
            # Otherwise, default to stored or newly initialized states
            output, (final_hidden_state, final_cell_state) = self.lstm(obs, self.hidden_cell)

        # Store the states for future use, can be reset externally when needed
        self.hidden_cell = (final_hidden_state, final_cell_state)
        return final_hidden_state[-1]

    def reset_states(self):
        # Call this method to reset states
        self.hidden_cell = None
# if __name__ == '__main__':
#     spac = gym.spaces.Box(low=0, high=2, shape=(1,))
#     print(bool(spac.sample() > 10))
