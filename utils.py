from __future__ import annotations

from pydogfight import Dogfight2dEnv
from pydogfight.policy import Policy, ManualPolicy, MultiAgentPolicy
from tqdm import tqdm
import pybts
import pydogfight
import os


def evaluate(env: Dogfight2dEnv, num_episodes: int, policy: Policy):
    env.render_info = {
        **env.render_info,
        **env.game_info
    }

    for _ in tqdm(range(num_episodes), desc='Dogfight Round'):
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


def create_bt_policy(env: Dogfight2dEnv, agent_name: str, filepath: str, track: bool, version: str,
                     track_interval: int):
    if agent_name in env.options.red_agents:
        agent_color = 'red'
    else:
        agent_color = 'blue'

    filename = os.path.basename(filepath).replace('.xml', '')
    tree = pybts.Tree(
            root=pydogfight.policy.BTPolicyBuilder().build_from_file(filepath),
            name=os.path.join(version, filename, agent_name),
            context={
                'agent_name' : agent_name,
                'agent_color': agent_color,
                'version'    : version,
                'filename'   : filename
            }
    )

    policy = pydogfight.policy.BTPolicy(
            env=env,
            tree=tree,
            agent_name=agent_name,
            update_interval=env.options.policy_interval
    )

    if track:
        board = pybts.Board(tree=policy.tree, log_dir='logs')
        board.clear()
        _last_track_time = env.time

        def on_post_tick(t):
            nonlocal _last_track_time
            if env.time - _last_track_time >= track_interval:
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
