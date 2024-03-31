from __future__ import annotations
import pydogfight
import pybts
from pydogfight import Dogfight2dEnv
from pydogfight.policy import BTPolicyBuilder, BTPolicy
import json
from tqdm import tqdm
import os

RENDER_MODE = 'human'
# RENDER_MODE = 'rgb_array'
BT_BOARD_TRACK = True
MAX_ROUND = 100  # 对战轮次

options = pydogfight.Options()
options.delta_time = 0.05
options.simulation_rate = 50
options.update_interval = 0.3


def create_greedy_policy(env: Dogfight2dEnv, agent_name: str):
    return pydogfight.policy.GreedyPolicy(env=env, agent_name=agent_name, update_interval=options.policy_interval)


def create_bt_greedy_policy(env: Dogfight2dEnv, agent_name: str, filepath: str):
    filename = os.path.basename(filepath).replace('.xml', '')
    tree = pybts.Tree(
            root=pydogfight.policy.BTPolicyBuilder().build_from_file(filepath),
            name=os.path.join(agent_name, filename))
    policy = pydogfight.policy.BTPolicy(
            env=env,
            tree=tree,
            agent_name=agent_name,
            update_interval=options.policy_interval
    )

    if BT_BOARD_TRACK:
        board = pybts.Board(tree=policy.tree, log_dir='logs')
        tree.add_post_tick_handler(lambda t: board.track({
            'env_time': env.time,
            **env.render_info,
            'red_1'   : env.get_agent('red_1').to_dict(),
        }))
        board.clear()
    return policy


def policy_main():
    # options.aircraft_radar_radius = options.game_size[0]
    env = Dogfight2dEnv(options=options, render_mode=RENDER_MODE)

    policy = pydogfight.policy.MultiAgentPolicy(
            # ManualPolicy(env=env, control_agents=options.agents, update_interval=0),
            # create_bt_greedy_policy(env=env, agent_name=options.red_agents[0],
            #                         filepath='./policies/bt_greedy_policy_default.xml'),
            # create_bt_greedy_policy(
            #         env=env,
            #         agent_name=options.red_agents[0],
            #         filepath='./policies/follow_route.xml'),
            # create_bt_greedy_policy(env=env, agent_name=options.red_agents[0],
            #                         filepath='./policies/bt_greedy_chatgpt_v1.xml'),
            # create_bt_greedy_policy(env=env, agent_name=options.blue_agents[0],
            #                         filepath='./policies/bt_greedy_chatgpt_v2.xml'),
            # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
            # GreedyPolicy(env=env, agent_name=options.red_agents[0], delta_time=1),
            # create_greedy_policy(env=env, agent_name=options.blue_agents[0]),
            create_bt_greedy_policy(
                    env=env,
                    agent_name=options.blue_agents[0],
                    filepath='./policies/follow_route.xml'),
    )

    win_count = {
        'red' : 0,
        'blue': 0,
        'draw': 0
    }

    env.render_info['red_wins'] = win_count['red']
    env.render_info['blue_wins'] = win_count['blue']
    env.render_info['draw'] = win_count['draw']

    for i in tqdm(range(MAX_ROUND)):
        obs, info = env.reset()
        policy.reset()
        if not env.isopen:
            break
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
                break
            if env.should_update():
                env.update()
            if env.should_render():
                env.render()

    with open('policy_demo_result.json', 'w') as f:
        json.dump(win_count, f)


if __name__ == '__main__':
    policy_main()
    # py_trees.display.render_dot_tree(root)
