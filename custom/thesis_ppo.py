from __future__ import annotations
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn.modules.module import T
import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy,
    MultiInputActorCriticPolicy
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from typing import Generator, NamedTuple
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    SelfCategoricalDistribution,
    Categorical
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class ThesisRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class ThesisRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """
    action_masks: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        self.action_mask_dim = 1
        if isinstance(action_space, spaces.Discrete):
            # Action is an int
            self.action_mask_dim = action_space.n
        super().__init__(
                buffer_size=buffer_size,
                observation_space=observation_space,
                action_space=action_space,
                device=device,
                gae_lambda=gae_lambda,
                gamma=gamma,
                n_envs=n_envs
        )

    def reset(self) -> None:
        super().reset()
        self.action_masks = np.zeros((self.buffer_size, self.n_envs, self.action_mask_dim), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            action_mask: np.ndarray | None = None
    ) -> None:
        if action_mask is not None:
            action_mask = action_mask.reshape((self.n_envs, self.action_mask_dim))
            self.action_masks[self.pos] = np.array(action_mask)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[ThesisRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                'action_masks'
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None,
    ) -> ThesisRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.action_masks[batch_inds]
        )
        return ThesisRolloutBufferSamples(*tuple(map(self.to_torch, data)))


def thesis_make_proba_distribution(
        action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = { }

    if isinstance(action_space, spaces.Box):
        cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(int(action_space.n), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(list(action_space.nvec), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(
                action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
                "Error: probability distribution, not implemented for action space"
                f"of type {type(action_space)}."
                " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


class ThesisActorCriticPolicy(ActorCriticPolicy):
    """
    自定义
    """

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor, action_mask: th.Tensor | None = None) -> Tuple[
        th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask=action_mask)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def evaluate_values(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)
        return values

    def forward(self, obs: th.Tensor, deterministic: bool = False, action_mask: np.ndarray | None = None) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor]:
        """
                Forward pass in all the networks (actor and critic)

                :param obs: Observation
                :param deterministic: Whether to sample or use deterministic actions
                :param action_mask: 动作掩码
                :return: action, value and log probability of the action
                """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask=action_mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, action_mask: np.ndarray | None = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            action_dist = self.action_dist.proba_distribution(action_logits=mean_actions)
            if action_mask is not None:
                # Apply the action mask by setting the probability of masked actions to zero
                action_logits = action_dist.distribution.logits
                # print('_get_action_dist_from_latent', action_logits.shape, action_mask.shape)
                # action_mask = action_mask.reshape((1, -1))
                action_logits[action_mask == 0] = -1 * 1e8
                # action_logits = action_logits + (th.tensor(action_mask).to(action_logits.device) - 1) * 1e8
                action_dist.distribution.logits = action_logits
            return action_dist
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def get_distribution(self, obs: PyTorchObs, action_mask: np.ndarray | None = None) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        # Get the base action distribution
        action_dist = self._get_action_dist_from_latent(latent_pi, action_mask=action_mask)
        return action_dist

    def _predict(self, observation: PyTorchObs, deterministic: bool = False,
                 action_mask: np.ndarray | None = None) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation, action_mask=action_mask).get_actions(deterministic=deterministic)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            action_mask: np.ndarray | None = None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                    "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                    "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                    "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                    "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                    "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic, action_mask=action_mask)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]


class ThesisPPO(PPO):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy"       : ThesisActorCriticPolicy,
        "CnnPolicy"       : ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            action_mask: np.ndarray | None = None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        assert isinstance(self.policy, ThesisActorCriticPolicy)
        return self.policy.predict(observation, state, episode_start, deterministic, action_mask=action_mask)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)  # 这行代码将策略模型切换到训练模式，这会影响如批标准化（batch norm）和dropout等操作。
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)  # 更新优化器的学习率
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator] # 根据当前训练进度计算clip范围。
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                    self._current_progress_remaining)  # type: ignore[operator] # 可选的clip范围（针对值函数）

        # 初始化存储损失值的列表。
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # 进行多个训练epoch。
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            # 在回合缓冲区上进行完整的遍历
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                assert isinstance(rollout_data, ThesisRolloutBufferSamples)
                actions = rollout_data.actions
                action_masks = rollout_data.action_masks
                # 处理离散动作空间
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    # 重采样噪声矩阵
                    self.policy.reset_noise(self.batch_size)

                assert isinstance(self.policy, ThesisActorCriticPolicy)

                # 评估动作
                values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions,
                        action_mask=action_masks)
                values = values.flatten()
                # Normalize advantage

                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    # 归一化优势
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 计算策略损失
                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging 记录日志
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # 计算值损失
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # 计算熵损失
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # 总损失
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    # 近似KL散度
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                # 早停
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                # 优化步骤
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
        # 更新计数器并记录日志
        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        #
        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)


class ThesisPPOOnlyValue(ThesisPPO):
    """
    PPO 只训练Value网络
    """

    # def train(self) -> None:
    #     """
    #     Update policy using the currently gathered rollout buffer.
    #     """
    #     assert isinstance(self.policy, ThesisActorCriticPolicy)
    #
    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(True)
    #     # Update optimizer learning rate
    #     self._update_learning_rate(self.policy.optimizer)
    #     pg_losses, value_losses = [], []
    #     continue_training = True
    #     # train for n_epochs epochs
    #     for epoch in range(self.n_epochs):
    #         # Do a complete pass on the rollout buffer
    #         for rollout_data in self.rollout_buffer.get(self.batch_size):
    #             values = self.policy.evaluate_values(rollout_data.observations)
    #             values = values.flatten()
    #             values_pred = values
    #             # Value loss using the TD(gae_lambda) target
    #             value_loss = F.mse_loss(rollout_data.returns, values_pred)
    #             value_losses.append(value_loss.item())
    #             loss = self.vf_coef * value_loss
    #             # Optimization step
    #             self.policy.optimizer.zero_grad()
    #             loss.backward()
    #             # Clip grad norm
    #             th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #             self.policy.optimizer.step()
    #
    #         self._n_updates += 1
    #         if not continue_training:
    #             break


class ThesisPPOOnlyAction(ThesisPPO):
    """
    PPO 只训练动作网络
    """

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                _, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # if self.clip_range_vf is None:
                #     # No clipping
                #     values_pred = values
                # else:
                #     # Clip the difference between old and new value
                #     # NOTE: this depends on the reward scaling
                #     values_pred = rollout_data.old_values + th.clamp(
                #             values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                #     )
                # # Value loss using the TD(gae_lambda) target
                # value_loss = F.mse_loss(rollout_data.returns, values_pred)
                # value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
