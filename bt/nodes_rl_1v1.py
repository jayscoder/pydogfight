import typing

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.policies import ActorCriticPolicy
from bt.nodes_rl import *
from pydogfight import Dogfight2dEnv
from pydogfight.utils.obs_utils import ObsUtils

# n_epochs（每次更新时优化代理损失的迭代次数）:
#
# 较少的迭代次数（如3-10次）通常足够大多数任务，可以在学习速度和算力消耗之间取得平衡。对于快速变化或较为简单的环境，可能更倾向于较少的迭代次数。
# 在你的场景中，考虑到对战的复杂性，可以从5次开始试验，然后根据结果调整。
# n_steps（每次更新为每个环境执行的步数）:
#
# 较多的步数意味着在每次更新前，代理将从环境中获得更多的经验。这有助于学习更稳定，但也可能增加对计算资源的需求。
# 鉴于每局对战平均时长为10分钟，你可以设置较高的步数，比如2048或4096步，以确保足够的探索和有效的梯度估计。
# batch_size（小批量的大小）:
#
# 小批量的大小影响梯度估计的噪声和计算效率。一般来说，较大的批量可以提高训练的稳定性，但计算成本也更高。
# 对于中等规模的训练环境，一般可以从64或128开始。这个值需要根据你的具体硬件配置进行调整，以避免内存溢出。

class RLGoToLocation1V1(RLGoToLocation):
    def rl_model_args(self) -> dict:
        attrs = {
            'gamma'        : 0.995,
            'policy_kwargs': {
                'features_extractor_class' : FeatureExtractor1V1,
                'features_extractor_kwargs': {
                    'features_dim': 128
                }
            }
        }

        if 'PPO' in self.algo:
            attrs.update({
                'n_steps'   : 120,
                'batch_size': 32,
                'n_epochs'  : 10,
            })
        elif 'SAC' in self.algo:
            attrs.update({
                'learning_starts': 120,
                'batch_size'     : 32,
                'train_freq'     : (1, "step"),
                'gradient_steps' : 1
            })

        return attrs


class RLFireAndGoToLocation1V1(RLFireAndGoToLocation):
    def rl_model_args(self) -> dict:
        attrs = {
            'gamma'        : 0.995,
            'policy_kwargs': {
                'features_extractor_class' : FeatureExtractor1V1,
                'features_extractor_kwargs': {
                    'features_dim': 128
                }
            }
        }

        if 'PPO' in self.algo:
            attrs.update({
                'n_steps'   : 120,
                'batch_size': 32,
                'n_epochs'  : 10,
            })
        elif 'SAC' in self.algo:
            attrs.update({
                'learning_starts': 120,
                'batch_size'     : 32,
                'train_freq'     : (1, "step"),
                'gradient_steps' : 1
            })
        return attrs


class RLCondition1V1(RLCondition):
    # def rl_observation_space(self) -> gym.spaces.Space:
    #     return gym.spaces.Box(low=0, high=1, shape=(1,))
    def rl_model_args(self) -> dict:
        attrs = {
            'gamma'        : 0.995,
            'policy_kwargs': {
                'features_extractor_class' : FeatureExtractor1V1,
                'features_extractor_kwargs': {
                    'features_dim': 128
                }
            }
        }
        if 'PPO' in self.algo:
            attrs.update({
                'n_steps'   : 12,
                'batch_size': 4,
                'n_epochs'  : 5,
            })
        elif 'SAC' in self.algo:
            attrs.update({
                'learning_starts': 12,
                'batch_size'     : 64,
                'train_freq'     : (1, "step"),
                'gradient_steps' : 1
            })
        return attrs

class FeatureExtractor1V1(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super(FeatureExtractor1V1, self).__init__(observation_space, features_dim)
        # 添加一个线性层用来处理提取的特征
        self.fc = nn.Linear(1 + 1 + 3 + 3 * ObsUtils.WATCH_MISSILES, features_dim)

    def forward(self, observations: torch.Tensor):
        self_can_fire_missile = observations[:, 0:1, 5]  # 自己能不能发射导弹
        enemy_can_fire_missile = observations[:, 1:2, 5]  # 敌人能不能发射导弹
        enemy_rel_polar_waypoint = observations[:, 1:2, 2:5]  # 敌人相对于自己的极坐标
        missile_rel_polar_waypoint = observations[:, 2:, 2:5]  # 敌人发射的导弹的极坐标

        combined_features = torch.cat([
            self_can_fire_missile,  # 增加一个维度以匹配其他特征的形状
            enemy_can_fire_missile,
            enemy_rel_polar_waypoint.reshape(observations.shape[0], -1),
            missile_rel_polar_waypoint.reshape(observations.shape[0], -1)  # 展平导弹的极坐标特征
        ], dim=1)

        # 将特征向量通过一个全连接层进行进一步处理
        processed_features = self.fc(combined_features)

        return processed_features
