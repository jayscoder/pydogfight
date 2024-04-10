from __future__ import annotations

from pydogfight import Dogfight2dEnv, Options
from pydogfight.policy import Policy, ManualPolicy, MultiAgentPolicy, BTPolicy
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


class BTManager:
    """用来做训练/测试，保存运行的数据"""

    def __init__(self, folder: str, env: Dogfight2dEnv, builder: pybts.Builder, track: int = -1):
        self.env = env
        self.runtime = now_str()

        self.folder = folder
        self.folder_runtime = os.path.join(folder, self.runtime)

        self.shell = 'python ' + ' '.join(sys.argv)
        self.builder = builder
        self.track = track

        os.makedirs(self.folder_runtime, exist_ok=True)
        self.write('options.json', self.env.options.to_dict())
        self.write('shell.sh', self.shell)
        self.write('builder.json', self.builder.repo_desc)

        self.policies: list[Policy] = []

    def add_bt_policy(
            self,
            agent_name: str,
            filepath: str,
            tree_name_suffix: str = ''
    ):
        env = self.env
        tree = pydogfight.policy.DogfightTree(
                env=env,
                agent_name=agent_name,
                root=self.builder.build_from_file(filepath),
                name=agent_name + tree_name_suffix,
                context={
                    'filename': self.builder.get_relative_filename(filepath=filepath),
                    'filepath': self.builder.find_filepath(filepath=filepath),
                    'folder'  : self.folder,
                    'runtime' : self.runtime,  # 执行时间
                }
        )

        policy = BTPolicy(
                env=env,
                tree=tree,
                agent_name=agent_name,
                update_interval=env.options.policy_interval
        )

        tree.setup(builder=self.builder)

        self.write(f'{agent_name}.xml', '\n'.join([f'<!--{filepath}-->', self.bt_to_xml(tree.root)]))
        render_node(tree.root, os.path.join(self.folder_runtime, f'{agent_name}.svg'))
        board = pybts.Board(tree=policy.tree, log_dir=self.folder_runtime)
        board.clear()

        def before_env_reset(env: Dogfight2dEnv):
            """保存环境的一些数据"""
            self.write(self.env_round_file(f'{agent_name}.xml'), self.bt_to_xml(tree.root))
            self.write(self.env_round_file(f'{agent_name}-context.json'), tree.context)
            board.track({
                'env_time': env.time,
                **env.render_info,
                **env.game_info,
                agent_name: env.get_agent(agent_name).to_dict(),
            })

        env.add_before_reset_handler(before_env_reset)

        if self.track >= 0:
            track_throttle = Throttle(duration=self.track)

            def on_tree_post_tick(t):
                if track_throttle.should_call(env.time):
                    board.track({
                        'env_time': env.time,
                        **env.render_info,
                        **env.game_info,
                        # **policy.tree.context,
                        # agent_name: env.get_agent(agent_name).to_dict(),
                    })

            def on_tree_reset(t):
                track_throttle.reset()

            tree.add_post_tick_handler(on_tree_post_tick)
            tree.add_reset_handler(on_tree_reset)

        self.policies.append(policy)

    def update_render_info(self):
        self.env.render_info = {
            **self.env.render_info,
            'time': int(self.env.time),
            **self.env.game_info,
        }

        for agent in self.env.battle_area.agents:
            self.env.render_info[f'{agent.name}_destroyed'] = agent.destroyed_count
            self.env.render_info[f'{agent.name}_missile_hit_self'] = agent.missile_hit_self_count
            self.env.render_info[f'{agent.name}_missile_hit_enemy'] = agent.missile_hit_enemy_count
            self.env.render_info[f'{agent.name}_missile_missile_miss'] = agent.missile_miss_count
            self.env.render_info[f'{agent.name}_return_home'] = agent.return_home_count
            self.env.render_info[f'{agent.name}_fuel'] = agent.fuel
            self.env.render_info[f'{agent.name}_missile_count'] = agent.missile_count

            for policy in self.policies:
                if isinstance(policy, BTPolicy) and policy.agent_name == agent.name:
                    rl_reward = policy.tree.context['rl_reward']
                    for k, v in rl_reward.items():
                        self.env.render_info[f'{policy.agent_name}_reward_{k}'] = v

    def run(self, num_episodes: int):
        env = self.env

        self.update_render_info()

        policy = MultiAgentPolicy(policies=self.policies)

        for i in tqdm(range(num_episodes), desc='Dogfight Round'):
            if not env.isopen:
                break
            env.reset()

            while env.isopen:
                policy.take_action()
                policy.put_action()
                info = env.gen_info()

                if info['terminated'] or info['truncated']:
                    # 在terminated之后还要再触发一次行为树，不然没办法将最终奖励给到行为树里的节点
                    # 所以这个判断在环境update之前，不能放在环境update之后
                    self.update_render_info()
                    break

                if env.should_update():
                    env.update()

                if env.should_render():
                    self.update_render_info()
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
        path = os.path.join(self.folder_runtime, path)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(content, str):
                f.write(content)
            else:
                pybts.utility.json_dump(content, f, indent=4, ensure_ascii=False)

    def delete(self, path: str):
        path = os.path.join(self.folder_runtime, path)
        if os.path.exists(path):
            os.remove(path)

    def bt_to_xml(self, node: pybts.behaviour.Behaviour):
        return pybts.utility.bt_to_xml(node,
                                       ignore_children=False,
                                       ignore_to_data=True,
                                       ignore_attrs=['id', 'name', 'status', 'tag', 'type', 'blackboard',
                                                     'feedback_messages',
                                                     'children_count'])


def create_manager(
        folder: str,
        policy_path: dict,  # 策略的文件，key是agent_name
        options=Options(),
        render: bool = False,
        track: int = -1,
):
    """

    Args:
        folder:
        policy_path: 策略路径字典，key是战队颜色或agent_name
        options:
        render:
        track:

    Returns:

    """
    builder = pydogfight.policy.BTPolicyBuilder(folders=[folder, 'scripts'])
    render_mode = 'human' if render else 'rgb_array'  # 是否开启可视化窗口
    env = Dogfight2dEnv(options=options, render_mode=render_mode)
    env.reset()

    manager = BTManager(
            folder=folder,
            env=env,
            builder=builder,
            track=track)

    for agent_name in options.red_agents:
        manager.add_bt_policy(
                agent_name=agent_name,
                filepath=policy_path.get(agent_name, policy_path['red']),
        )

    for agent_name in options.blue_agents:
        manager.add_bt_policy(
                agent_name=agent_name,
                filepath=policy_path.get(agent_name, policy_path['blue']),
        )

    return manager
