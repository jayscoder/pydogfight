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

## 问题

### 1. "dot" not found in path.

可能是电脑上没有安装graphviz，请参照如下网址安装
https://graphviz.org/download/


