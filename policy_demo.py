from __future__ import annotations
import math

import py_trees.display
import stable_baselines3.common.callbacks

from pydogfight.core.actions import Actions
from pydogfight import *
import numpy as np
from queue import Queue
from collections import defaultdict
import random
from pydogfight.policy import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from py_trees import visitors


class BTGreedyPolicyVisitor(visitors.VisitorBase):

    def run(self, behaviour: behaviour.Behaviour) -> None:
        print('Behaviour:', behaviour, behaviour.status)


def policy_main():
    options = Options()
    options.delta_time = 0.1
    options.simulation_rate = 40
    options.update_interval = 1
    # options.aircraft_radar_radius = options.game_size[0]
    env = Dogfight2dEnv(options=options, render_mode='human')

    # 每隔3s做一次操作
    # red_policy = GreedyPolicy(env=env, agent_name=options.red_agents[0], delta_time=5)
    visitor = BTGreedyPolicyVisitor()
    policy = MultiAgentPolicy(
            env=env,
            policies=[
                # ManualPolicy(env=env, control_agents=options.agents, update_interval=0),
                BTPolicy(env=env, root=BTGreedyBuilder().build_from_xml_text(BT_GREEDY_XML),
                         agent_name=options.red_agents[0], update_interval=1,
                         visitor=visitor,
                         ),
                # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
                # GreedyPolicy(env=env, agent_name=options.red_agents[0], delta_time=1),
                GreedyPolicy(env=env, agent_name=options.blue_agents[0], update_interval=1)
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

    env.reset()
    policy.reset()

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
    # policy_main()
    root = BTGreedyBuilder().build_from_xml_text(BT_GREEDY_XML)
    root.status = py_trees.common.Status.SUCCESS
    print(py_trees.display.ascii_tree(root, show_status=True))
    print(py_trees.display.unicode_tree(root, show_status=True))
    # py_trees.display.render_dot_tree(root)
