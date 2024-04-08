from __future__ import annotations

import os.path
from typing import Any, Dict

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from pydogfight import *
from pydogfight.policy import Policy, ManualPolicy
from pydogfight.wrappers import AgentWrapper
import json
import pybts
import pydogfight
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description="PPO Training and Testing")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'teacher'],
                    help='Mode to run the script in: train, test, or teacher')
parser.add_argument('--render_mode', type=str, default='rgb_array', choices=['rgb_array', 'human'],
                    help='Render Mode')
parser.add_argument('--red_bt', type=str, default='policies/bt_greedy.xml', help='Path to the red policy')
parser.add_argument('--blue_bt', type=str, default='policies/bt_greedy.xml', help='Path to the blue policy')
parser.add_argument('--track_name',
                    type=str,
                    default='main',
                    help='BT Track Name')
parser.add_argument('--track_interval',
                    type=int,
                    default=10,
                    help='每隔几s track一次')
parser.add_argument('--n',
                    type=int,
                    default=100,
                    help='循环轮次')
args = parser.parse_args()
print('PYDOGFIGHT', args)

options = Options()
options.train = args.mode == 'train'
N = args.n
BT_BOARD_TRACK_NAME = args.track_name
env = Dogfight2dEnv(options=options, render_mode=args.render_mode)


def create_bt_policy(env: Dogfight2dEnv, agent_name: str, filepath: str, track: bool):
    filename = os.path.basename(filepath).replace('.xml', '')
    tree = pybts.Tree(
            root=pydogfight.policy.BTPolicyBuilder().build_from_file(filepath),
            name=os.path.join(BT_BOARD_TRACK_NAME, agent_name, filename))
    policy = pydogfight.policy.BTPolicy(
            env=env,
            tree=tree,
            agent_name=agent_name,
            update_interval=options.policy_interval
    )

    if track:
        board = pybts.Board(tree=policy.tree, log_dir='logs')
        board.clear()
        _last_track_time = env.time

        def on_post_tick(t):
            nonlocal _last_track_time
            if env.time - _last_track_time >= args.track_interval:
                _last_track_time = env.time
                board.track({
                    'env_time': env.time,
                    **env.render_info,
                    agent_name: env.get_agent(agent_name).to_dict(),
                })

        def on_reset(t):
            nonlocal _last_track_time
            _last_track_time = 0

        tree.add_post_tick_handler(on_post_tick)
        tree.add_reset_handler(on_reset)

    return policy

def main():
    policy = pydogfight.policy.MultiAgentPolicy(
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
            create_bt_policy(
                    env=env,
                    agent_name=options.red_agents[0],
                    filepath=args.red_bt,
                    track=True
            ),
            create_bt_policy(
                    env=env,
                    agent_name=options.blue_agents[0],
                    filepath=args.blue_bt,
                    track=True
            ),
    )

    env.render_info = {
        **env.render_info,
        **env.game_info
    }

    for _ in tqdm(range(N), desc='Dogfight Round'):
        if not env.isopen:
            break
        env.reset()
        policy.reset()

        while env.isopen:
            policy.take_action()
            policy.put_action()
            info = env.gen_info()

            if info['terminated'] or info['truncated']:
                env.render_info = {
                    **env.render_info,
                    **env.game_info
                }
                break
            if env.should_update():
                env.update()
            if env.should_render():
                env.render()


if __name__ == '__main__':
    main()
