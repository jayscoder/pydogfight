# pydogfight

基于Gym开发的2D空战仿真框架

## 运行

```shell
python main.py scripts/v2/ppo.yaml --render --output-dir=outputs/ppo_v2  
```

## 配置文件

scripts/v2/ppo.yaml

```shell
output_dir: output/v2/ppo # 输出目录
render: false # 是否渲染
num_episodes: 1000 # 运行轮数
track: 60 # 是否捕获行为树运行数据
policy:
    red: scripts/v2/rl.xml # 红方行为树
    blue: scripts/v2/greedy.xml # 蓝方行为树
options:
    train: true # 是否开启训练
    red_agents: ['red'] # 红方agents
    blue_agents: ['blue'] # 蓝方agents
#     indestructible: true # 是否开启无敌模式
    collision_scale: 1.5 # 碰撞倍数，越大越容易发生碰撞
context: # 行为树环境变量，可以在行为树中通过{{}}来传递
    models_dir: 'models/v2/ppo' # 模型目录
    rl_algo: PPO
---
# 后面的配置会继承前面的
num_episodes: 100
options:
    train: false
    indestructible: false
    collision_scale: 1.5
```

## 强化学习结合的行为树

行为树框架：https://github.com/wangtong2015/pybts

具体请看scripts中写的行为树xml和 pydogfight/policy/bt/ 中的代码

## 行为树节点定义

具体请看

- bt/ ：这里放自定义的节点
- pydogfight/policy/bt/ 这里是默认提供的贪心节点
- pydogfight/policy/bt/rl 这里是默认提供的强化学习节点

> 上述节点均默认注册

## 强化学习

approx_kl: 近似Kullback-Leibler散度，即策略更新前后的变化量。数值为0.028604157，表示在连续更新中，新策略与旧策略之间的差异是适中的。如果这个值过高，可能表示策略更新过于激进，可能导致训练不稳定。

clip_fraction: 这是PPO算法中使用的剪裁比例，其值为0.138，意味着大约13.8%的梯度被剪裁。这是为了避免更新步骤过大，保持训练的稳定性。

clip_range: 剪裁范围，这里是0.2。这个范围决定了策略更新的最大步长，防止更新过大而导致训练不稳定。

entropy_loss: 熵损失，值为-2.77。熵损失是用来衡量策略的随机性，高熵意味着策略的随机性更高，有助于探索；而低熵则意味着策略趋于确定性。负值表示熵在减少，策略逐渐趋向确定性。

explained_variance: 解释方差，值为-0.156。这个指标衡量的是模型预测的值函数和实际回报之间的一致性。理想情况下，这个值越接近1越好。这里的负值表明模型的预测和实际情况差异较大。

learning_rate: 学习率，这里为0.0003。学习率决定了权重更新的步长，对学习过程的速度和质量有直接影响。

loss: 总损失，值为-0.027。这是模型在训练过程中的总体损失，是优化的直接目标。

n_updates: 更新次数，290次。表示到目前为止，模型参数已经被更新了290次。

policy_gradient_loss: 策略梯度损失，值为-0.0213。这个损失反映了策略梯度优化器的性能，是模型学习策略的直接反馈。

std: 策略的标准差，值为0.964。这个指标反映了采取行动的随机性，较大的值表明采取的行动较为多样。

value_loss: 值函数损失，值为0.00489。这是值函数预测与实际回报之间差异的量度，用于优化模型的值函数预测。

## 环境

战场：1v1战机对抗
策略更新时间间隔：1s
一局对战最长时长：30min
一局对战平均时长：10min

状态空间：

- shape=(15, 8)
- 每一行都是一个实体的数据（飞机、导弹），包括坐标、发射状态、速度等

动作空间：

- shape=(3, )
- 1：动作类型
  - 0：无
  - 1：go to location
  - 2: fire missile
- 2、3：动作指定坐标（x,y）

框架：stable_baseline3

强化学习策略：PPO

## 强化学习算法

### DQN

- **类型**：用于离散动作空间。
- **原理**：DQN 是基于Q学习的一种算法，使用深度神经网络来近似Q函数，即动作-价值函数，它预测在给定状态和动作下的期望回报。DQN 引入了经验回放（replay buffer）和目标Q网络这两个关键技术来稳定训练和避免发散。
- **特点**：简单，易于实现，广泛应用于具有离散动作空间的问题，但不适用于连续动作空间。



### SAC（Soft Actor-Critic）

- **类型**：用于连续动作空间。
- **原理**：SAC 是一个基于Actor-Critic架构的算法，它结合了深度学习和强化学习的技术。SAC的核心在于最大化预期回报和熵（动作的随机性），这样可以鼓励探索。SAC通常具有更好的样本效率和更稳定的训练性能。
- **特点**：可以处理连续动作问题，通过熵正则化促进探索，训练相对稳定。



### DDPG（Deep Deterministic Policy Gradient）

- **类型**：用于连续动作空间。
- **原理**：DDPG是一个Actor-Critic算法，结合了Q学习的思想和策略梯度方法。它使用一个确定性的策略（Actor）来选择动作，和一个值函数（Critic）来评估这个动作。DDPG也使用了经验回放和目标网络技术，类似于DQN。
- **特点**：适用于连续动作空间，可以学习确定性策略，训练较为稳定，但可能面临探索不足的问题。



###  TD3（Twin Delayed DDPG）

- **类型**：用于连续动作空间。
- **原理**：TD3是对DDPG的一个改进，它引入了双Critic架构来减少值函数估计的过优化问题，同时通过延迟策略更新和目标策略平滑来进一步提高算法的稳定性。
- **特点**：相比DDPG，TD3在许多环境中表现出更好的性能和更高的样本效率，减少了过优化问题。






## 问题

### 1. "dot" not found in path.

可能是电脑上没有安装graphviz，请参照如下网址安装
https://graphviz.org/download/

