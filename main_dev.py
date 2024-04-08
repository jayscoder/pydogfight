from __future__ import annotations

import os.path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from py_trees.common import Status

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
from stable_baselines3.common.utils import (
    explained_variance, get_schedule_fn, safe_mean, obs_as_tensor,
    configure_logger
)
import jinja2

parser = argparse.ArgumentParser(description="PPO Training and Testing")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'teacher'],
                    help='Mode to run the script in: train, test, or teacher')
parser.add_argument('--render_mode', type=str, default='rgb_array', choices=['rgb_array', 'human'],
                    help='Render Mode')
parser.add_argument('--track_name',
                    type=str,
                    default='main_rl',
                    help='BT Track Name')
args = parser.parse_args()
print('PYDOGFIGHT', args)

options = Options()
options.delta_time = 0.5  # 每次更新的间隔
options.self_side = 'red'
options.simulation_rate = 100
options.train = args.mode == 'train'
TEST_N = 100
BT_BOARD_TRACK_NAME = args.track_name

env = Dogfight2dEnv(options=options, render_mode=args.render_mode)


class PrintMsg(pybts.Action):
    def __init__(self, msg: str, name: str = ''):
        super().__init__(name=name)
        self.msg = msg

    def update(self) -> Status:
        print(self.name, self.msg)
        return Status.SUCCESS


def dev():
    root = pydogfight.policy.bt.rl.ppo.PPOSwitcher(
            path='models/ppo_switcher',
            children=[
                PrintMsg(msg='A1'),
                PrintMsg(msg='A2'),
                PrintMsg(msg='A3')
            ]
    )
    tree = pybts.Tree(root)
    policy = pydogfight.policy.bt.BTPolicy(
            tree=tree,
            env=env,
            agent_name='red_1',
    )

    env.reset()
    policy.reset()
    obs = env.gen_agent_obs(agent_name='red_1')
    model = root.model

    print(model.observation_space)
    model.policy.set_training_mode(False)
    obs_tensor = obs_as_tensor(np.expand_dims(obs, axis=0), root.model.device)
    actions, values, log_probs = root.model.policy(obs_tensor)
    print(root.__str__(), actions[0])


def dev_2():
    model = PPO(policy='MlpPolicy', env=env)
    model.learn(total_timesteps=100, progress_bar=True)


if __name__ == '__main__':
    # dev()
    print(jinja2.Template('color').render({ 'color': 'red' }))
