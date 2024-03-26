from __future__ import annotations
import math

import stable_baselines3.common.callbacks

from gym_dogfight.core.actions import Actions
from gym_dogfight import *
import numpy as np
from queue import Queue
from collections import defaultdict
import random
from gym_dogfight.policy import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def policy_main():
    options = Options()
    options.delta_time = 0.1
    options.simulation_rate = 40
    options.update_interval = 1
    env = Dogfight2dEnv(options=options, render_mode='human')
    obs, info = env.reset()

    # 每隔3s做一次操作
    # red_policy = GreedyPolicy(env=env, agent_name=options.red_agents[0], delta_time=5)

    policy = MultiAgentPolicy(
            env=env,
            policies=[
                # ManualPolicy(env=env, control_agents=options.red_agents, delta_time=0),
                # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
                GreedyPolicy(env=env, agent_name=options.red_agents[0], delta_time=5),
                GreedyPolicy(env=env, agent_name=options.blue_agents[0], delta_time=5)
            ],
    )

    win_count = {
        'red' : 0,
        'blue': 0,
        'draw': 0
    }

    env.render_info['red_wins'] = win_count['red']
    env.render_info['blue_wins'] = win_count['blue']
    env.render_info['draw'] = win_count['draw']

    while env.isopen:
        policy.select_action()
        policy.put_action()

        info = env.gen_info()

        if info['terminated'] or info['truncated']:
            if info['winner'] in win_count:
                win_count[info['winner']] += 1
                env.render_info['red_wins'] = win_count['red']
                env.render_info['blue_wins'] = win_count['blue']
                env.render_info['draw'] = win_count['draw']
            obs, info = env.reset()
            policy.reset()
        if env.should_update():
            env.update()
        if env.should_render():
            env.render()


if __name__ == '__main__':
    policy_main()
