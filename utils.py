from __future__ import annotations

from pydogfight import Dogfight2dEnv
from pydogfight.policy import Policy, ManualPolicy, MultiAgentPolicy
from tqdm import tqdm
import pybts
import pybts.rl
import pydogfight
import os
import sys
import time
from datetime import datetime
import json
from pybts.display import render_node


def now_str():
    return datetime.now().strftime("%m%d-%H%M%S")


class Throttle:
    def __init__(self, duration: float):
        self.last_time = -float('inf')
        self.duration = duration

    def reset(self):
        self.last_time = -float('inf')

    def should_call(self, time: float) -> bool:
        if time - self.last_time >= self.duration:
            self.last_time = time
            return True
        else:
            return False


class Manager:
    """用来做训练/测试，保存运行的数据"""

    ROOT = 'workspace'

    def __init__(self, env: Dogfight2dEnv, builder: pybts.Builder, bt_track: bool = False, bt_track_interval: int = 0):
        self.env = env
        self.runtime = now_str()
        self.folder = os.path.join(self.ROOT, self.runtime)
        self.shell = 'python ' + ' '.join(sys.argv)
        self.builder = builder
        self.bt_track = bt_track
        self.bt_track_interval = bt_track_interval

        os.makedirs(self.folder, exist_ok=True)
        self.write('options.json', self.env.options.to_dict())
        self.write('shell.sh', self.shell)
        self.write('builder.json', self.builder.repo_desc)

    def create_bt_policy(
            self,
            agent_name: str,
            filepath: str,
            tree_name_suffix: str = ''
    ):
        env = self.env

        if agent_name in env.options.red_agents:
            agent_color = 'red'
        else:
            agent_color = 'blue'

        tree = pybts.rl.RLTree(
                root=self.builder.build_from_file(filepath),
                name=agent_name + tree_name_suffix,
                context={
                    'agent_name' : agent_name,
                    'agent_color': agent_color,
                    'version'    : env.options.version,
                    'time'       : lambda: env.time,
                    'runtime'    : self.runtime  # 执行时间
                }
        )

        policy = pydogfight.policy.BTPolicy(
                env=env,
                tree=tree,
                agent_name=agent_name,
                update_interval=env.options.policy_interval
        )

        tree.setup(builder=self.builder,
                   env=env,
                   agent_name=agent_name,
                   actions=policy.actions)

        self.write(f'{agent_name}-bt.xml', '\n'.join([f'<!--{filepath}-->', self.bt_to_xml(tree.root)]))
        render_node(tree.root, os.path.join(self.folder, f'{agent_name}-bt.svg'))
        board = pybts.Board(tree=policy.tree, log_dir=self.folder)
        board.clear()

        def after_env_update(env: Dogfight2dEnv):
            tree.context['time'] = env.time

        def after_env_reset(env: Dogfight2dEnv):
            policy.reset()

        def before_env_reset(env: Dogfight2dEnv):
            """保存环境的一些数据"""
            self.write(self.env_round_file(f'{agent_name}-bt.xml'), self.bt_to_xml(tree.root))
            self.write(self.env_round_file(f'{agent_name}-bt-context.json'), tree.context)
            board.track({
                'env_time': env.time,
                **env.render_info,
                **env.game_info,
                agent_name: env.get_agent(agent_name).to_dict(),
            })

        env.add_after_update_handler(after_env_update)
        env.add_before_reset_handler(before_env_reset)
        env.add_after_reset_handler(after_env_reset)

        if self.bt_track:
            track_throttle = Throttle(duration=self.bt_track_interval)

            def on_tree_post_tick(t):
                if track_throttle.should_call(env.time):
                    board.track({
                        'env_time': env.time,
                        **env.render_info,
                        **env.game_info,
                        agent_name: env.get_agent(agent_name).to_dict(),
                    })

            def on_tree_reset(t):
                track_throttle.reset()

            tree.add_post_tick_handler(on_tree_post_tick)
            tree.add_reset_handler(on_tree_reset)

        return policy

    def create_policy_from_workspace(self, runtime: str):
        """
        从workspace的历史文件中构造策略
        """
        policies = []
        for agent_name in self.env.options.agents():
            bt_path = os.path.join(self.ROOT, runtime, f'{agent_name}-bt.xml')
            policies.append(self.create_bt_policy(agent_name=agent_name, filepath=bt_path))
        return MultiAgentPolicy(policies=policies)

    def evaluate(self, num_episodes: int, policy: Policy):
        env = self.env

        env.render_info = {
            **env.render_info,
            **env.game_info
        }

        for i in tqdm(range(num_episodes), desc='Dogfight Round'):
            if not env.isopen:
                break
            env.reset()

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

            self.write(self.env_round_file('env_info.json'), env.gen_info())
            self.write(self.env_round_file('game_info.json'), {
                **env.render_info,
                **env.game_info,
            })

            for agent in self.env.battle_area.agents:
                self.write(self.env_round_file(f'{agent.name}.json'), agent.to_dict())

            self.write(f'env.json', {
                **env.gen_info(),
                **env.render_info,
                **env.game_info,
            })
            self.write(f'run-{i}-{num_episodes}.txt', f'{i}/{num_episodes}')
            self.delete(f'run-{i - 1}-{num_episodes}.txt')

    def env_round_file(self, file: str):
        return os.path.join('env', str(self.env.game_info['round']), file)

    def write(self, path: str, content: str | dict) -> None:
        path = os.path.join(self.folder, path)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(content, str):
                f.write(content)
            else:
                pybts.utility.json_dump(content, f, indent=4, ensure_ascii=False)

    def delete(self, path: str):
        path = os.path.join(self.folder, path)
        if os.path.exists(path):
            os.remove(path)

    def bt_to_xml(self, node: pybts.behaviour.Behaviour):
        return pybts.utility.bt_to_xml(node,
                                       ignore_children=False,
                                       ignore_to_data=True,
                                       ignore_attrs=['id', 'name', 'status', 'tag', 'type', 'blackboard',
                                                     'feedback_messages',
                                                     'children_count'])
