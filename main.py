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

parser = argparse.ArgumentParser(description="PPO Training and Testing")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                    help='Mode to run the script in: train, test, or teacher')

parser.add_argument('--render-mode', type=str, default='rgb_array', choices=['rgb_array', 'human'],
                    help='Render Mode')
parser.add_argument('--red-bt', type=str, default='policies/bt_greedy.xml', help='Path to the red policy')
parser.add_argument('--blue-bt', type=str, default='policies/bt_greedy.xml', help='Path to the blue policy')
parser.add_argument('--track', action='store_true', help='是否要track')
parser.add_argument('--delta-time', type=float, default=0.01, help='options.delta_time')
parser.add_argument('--update-interval', type=float, default=1,
                    help='options.update_interval 每轮env更新的时间间隔（在一轮更新中会进行多次更新，更新次数=update_interval/delta_time）')
parser.add_argument('--policy-interval', type=float, default=0,
                    help='options.policy_interval 每次策略的处理间隔时长，0代表每次更新后都提供策略')

parser.add_argument('--track-interval',
                    type=int,
                    default=10,
                    help='每隔几s track一次')
parser.add_argument('--num-episodes',
                    type=int,
                    default=100,
                    help='循环轮次')
parser.add_argument('--version',
                    type=str,
                    default='v1',
                    help='版本')

args = parser.parse_args()
SHELLS = ' '.join(sys.argv[1:])
options = Options()
options.train = args.mode == 'train'
options.delta_time = args.delta_time
options.update_interval = args.update_interval
options.policy_interval = args.policy_interval
env = Dogfight2dEnv(options=options, render_mode=args.render_mode)
VERSION = args.version


def main():
    policy = MultiAgentPolicy(
            policies=[
                # pydogfight.policy.ModelPolicy(
                #         env=env, model=model,
                #         agent_name=options.red_agents[0],
                #         update_interval=options.policy_interval),
                # create_bt_model_policy(
                #         env=env, agent_name=options.red_agents[0],
                #         track=True
                # ),
                # ManualPolicy(env=env, control_agents=options.agents, update_interval=0),
                # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
                utils.create_bt_policy(
                        env=env,
                        agent_name=options.red_agents[0],
                        filepath=args.red_bt,
                        track=args.track,
                        version=args.version,
                        track_interval=args.track_interval
                ),
                utils.create_bt_policy(
                        env=env,
                        agent_name=options.blue_agents[0],
                        filepath=args.blue_bt,
                        track=args.track,
                        version=args.version,
                        track_interval=args.track_interval
                ),
            ]
    )
    utils.evaluate(
            env=env,
            num_episodes=args.num_episodes,
            policy=policy
    )


if __name__ == '__main__':
    main()
