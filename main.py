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

parser.add_argument('--train', action='store_true', help='是否开启训练模式')

parser.add_argument('--render-mode', type=str, default='rgb_array', choices=['rgb_array', 'human'],
                    help='渲染模式，rgb_array用来无窗口训练，human会展示可视化窗口')

parser.add_argument('--red-bt', type=str, default='scripts/bt_greedy.xml', help='红方行为树路径')
parser.add_argument('--blue-bt', type=str, default='scripts/bt_greedy.xml', help='蓝方行为树路径')
parser.add_argument('--track', action='store_true', help='是否要用pybts记录行为树运行数据')
parser.add_argument('--track-interval', type=int, default=10, help='每隔几s track一次')

parser.add_argument('--delta-time', type=float, default=options.delta_time, help='options.delta_time')
parser.add_argument('--update-interval', type=float, default=options.update_interval,
                    help='options.update_interval 每轮env更新的时间间隔（在一轮更新中会进行多次更新，更新次数=update_interval/delta_time）')
parser.add_argument('--policy-interval', type=float, default=options.policy_interval,
                    help='options.policy_interval 每次策略的处理间隔时长，0代表每次更新后都提供策略')

parser.add_argument('--num-episodes', type=int, default=100, help='对战场次')
parser.add_argument('--version', type=str, default=options.version, help='版本')
parser.add_argument('--indestructible', action='store_true', help='战机是否不可被摧毁')
parser.add_argument('--max-duration', type=int, default=options.max_duration,
                    help='一局对战最长时间，单位是s，默认是30分钟')
parser.add_argument('--simulation-rate', type=int, default=options.simulation_rate,
                    help='仿真的速率倍数，越大代表越快，update_interval内更新几次（仅在render_mode=human模式下生效）')
parser.add_argument('--collision-scale', type=float, default=options.collision_scale,
                    help='碰撞半径倍数，越大代表越容易触发碰撞')
args = parser.parse_args()

options.train = args.train
options.delta_time = args.delta_time
options.update_interval = args.update_interval
options.policy_interval = args.policy_interval
options.version = args.version
options.aircraft_indestructible = args.indestructible
options.max_duration = args.max_duration
options.simulation_rate = args.simulation_rate
options.collision_scale = args.collision_scale


def main():
    env = Dogfight2dEnv(options=options, render_mode=args.render_mode)
    env.reset()

    builder = pydogfight.policy.BTPolicyBuilder(folders=['scripts', 'workspace'])

    manager = utils.Manager(env=env, builder=builder, bt_track=args.track, bt_track_interval=args.track_interval)

    # policy = MultiAgentPolicy(
    #         policies=[
    #             # ManualPolicy(env=env, control_agents=options.agents, update_interval=0),
    #             # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
    #             manager.create_bt_policy(
    #                     agent_name=options.red_agents[0],
    #                     filepath='./scripts/manual_control.xml',
    #                     tree_name_suffix='-manual'
    #             ),
    #             manager.create_bt_policy(
    #                     agent_name=options.red_agents[0],
    #                     filepath=args.red_bt,
    #             ),
    #             manager.create_bt_policy(
    #                     agent_name=options.blue_agents[0],
    #                     filepath=args.blue_bt,
    #             ),
    #         ]
    # )
    policy = manager.create_policy_from_workspace(runtime='0410-024433')
    manager.evaluate(
            num_episodes=args.num_episodes,
            policy=policy
    )


if __name__ == '__main__':
    main()
