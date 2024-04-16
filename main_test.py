from stable_baselines3 import SAC
import gymnasium as gym
from bt.base import *
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.policies import ActorCriticPolicy


class V7FeatureExtractor1V1(BaseFeaturesExtractor):
    def __init__(self, observation_space, shared: nn.Module, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 添加一个线性层用来处理提取的特征
        self.shared = nn.Linear(in_features=1, out_features=1)

    def forward(self, observations: torch.Tensor):
        return self.shared(observations)


if __name__ == '__main__':
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    env.reset()
    env = ImgObsWrapper(env)
    model = PPO('MlpPolicy', env=env, verbose=1, policy_kwargs={
        'features_extractor_class' : V7FeatureExtractor1V1,
        'features_extractor_kwargs': {
            'features_dim': 128,
            'shared'      : nn.Linear(1, 1)
        }
    })
    print(model.policy.features_extractor.state_dict())
    model.save('main_test')
    del model
    model = PPO('MlpPolicy', env=env, verbose=1, policy_kwargs={
        'features_extractor_class' : V7FeatureExtractor1V1,
        'features_extractor_kwargs': {
            'features_dim': 128,
            'shared'      : nn.Linear(1, 1)
        }
    })
    # model = PPO.load('main_test')
    model.set_parameters('main_test')
    print(model.policy.features_extractor.state_dict())
