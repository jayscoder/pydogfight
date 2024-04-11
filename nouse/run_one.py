from __future__ import annotations

import os.path
import sys

from typing import Any, Dict

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from pydogfight import Dogfight2dEnv, Options
from pydogfight.policy import Policy, ManualPolicy, MultiAgentPolicy
from pydogfight.wrappers import AgentWrapper
import json
import pybts
import pydogfight
import argparse
from tqdm import tqdm
import utils

options = Options()

parser = argparse.ArgumentParser(description="Dogfight Training and Testing")

parser.add_argument('--output-dir', type=str, default='output/default', help='工作输出目录')

parser.add_argument('--train', action='store_true', help='是否开启训练模式')

parser.add_argument('--render', action='store_true', help='是否开启可视化窗口，在做强化学习训练的时候建议关闭来提高性能')

parser.add_argument('--red', type=str, default='scripts/greedy.xml', help='红方行为树路径，可以是相对于--folder的路径')
parser.add_argument('--blue', type=str, default='scripts/greedy.xml', help='蓝方行为树路径，可以是相对于--folder的路径')
parser.add_argument('--track', type=int, default=-1, help='每隔几秒用pybts track一次，-1表示不track')
# 想要开启pybts的可视化面板的话，执行 pybts --dir=scripts

parser.add_argument('--delta-time', type=float, default=options.delta_time, help='options.delta_time')
parser.add_argument('--update-interval', type=float, default=options.update_interval,
                    help='options.update_interval 每轮env更新的时间间隔（在一轮更新中会进行多次更新，更新次数=update_interval/delta_time）')
parser.add_argument('--policy-interval', type=float, default=options.policy_interval,
                    help='options.policy_interval 每次策略的处理间隔时长，0代表每次更新后都提供策略')
parser.add_argument('--num-episodes', type=int, default=1000, help='对战场次')
parser.add_argument('--indestructible', action='store_true',
                    help='战机是否开启无敌模式，在做强化学习训练的时候就不能靠战机是否被摧毁来获得奖励，需要靠导弹命中敌机来获得奖励')
parser.add_argument('--max-duration', type=int, default=options.max_duration,
                    help='一局对战最长时间，单位是s')
parser.add_argument('--simulation-rate', type=int, default=options.simulation_rate,
                    help='仿真的速率倍数，越大代表越快，update_interval内更新几次（仅在render_mode=human模式下生效）')
parser.add_argument('--collision-scale', type=float, default=options.collision_scale,
                    help='碰撞半径倍数，越大代表越容易触发碰撞')
parser.add_argument('--models-dir', type=str, default='models',
                    help='模型目录')

args = parser.parse_args()

def main():
    options.train = args.train
    options.delta_time = args.delta_time
    options.update_interval = args.update_interval
    options.policy_interval = args.policy_interval
    options.indestructible = args.indestructible
    options.max_duration = args.max_duration
    options.simulation_rate = args.simulation_rate
    options.collision_scale = args.collision_scale

    policy_path = { }

    for agent_name in options.red_agents:
        policy_path[agent_name] = args.red
    for agent_name in options.blue_agents:
        policy_path[agent_name] = args.blue

    manager = utils.create_manager(
            output_dir=args.output_dir,
            policy_path=policy_path,
            render=args.render,
            options=options,
            track=args.track,
            context={
                'models_dir': args.models_dir
            }
    )
    manager.run(args.num_episodes)


if __name__ == '__main__':
    main()
