from custom.thesis_ppo import *


class ThesisMixedPPO(ThesisPPO):
    """
    混合价值网络的PPO
    """

    def train(self, models: list[ThesisPPO] = None) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        policies 额外的参与训练的policy
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)  # 这行代码将策略模型切换到训练模式，这会影响如批标准化（batch norm）和dropout等操作。
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)  # 更新优化器的学习率
        for m in models:
            m.policy.set_training_mode(True)
            m._update_learning_rate(m.policy.optimizer)
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
                rollout_data
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
                for m in models:
                    assert isinstance(m.policy, ThesisActorCriticPolicy)
                    m.policy.evaluate_values()

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
