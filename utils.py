from __future__ import annotations

from pydogfight import Dogfight2dEnv, Options
from pydogfight.policy import Policy, ManualPolicy, MultiAgentPolicy, BTPolicy, DogfightTree
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
import bt
import yaml
from collections import defaultdict
from pydogfight.utils import get_torch_device

def now_str():
    return datetime.now().strftime("%m%d-%H-%M-%S")


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


TEMPLATE_CONFIG = {
    'title'   : '',
    'desc'    : '',
    'output'  : 'output/default',
    'episodes': 100,
    'track'   : 100,
    'policy'  : {
    },
    'options' : {
        'render': False
    },
    'context' : { }
}


def read_config(path: str):
    """读取配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        config = list(yaml.load_all(f, Loader=yaml.FullLoader))[0]
    if 'base' in config:
        config = merge_config(read_config(config['base']), config)

    if config['options'].get('title') is None and config.get('title') is not None:
        config['options']['title'] = config['title']

    return config


def merge_config(base: dict, config: dict):
    new_config = base.copy()
    for k in config:
        if k in ['options', 'context', 'policy']:
            if new_config.get(k) is None:
                new_config[k] = { }
            new_config[k].update(config.get(k) or { })
        else:
            new_config[k] = config[k]
    return new_config


class BTManager:
    """用来做训练/测试，保存运行的数据"""

    def __init__(self, config: dict, builder: pybts.Builder):
        self.title = config.get('title', config['output'])
        self.desc = config.get('desc', '')

        options = Options()
        if 'options' in config:
            options.load_dict(config['options'])

        options.validate()

        if options.device == 'auto':
            options.device = get_torch_device(options.device)

        env = Dogfight2dEnv(options=options)
        env.reset()
        self.config = config

        self.env = env
        self.runtime = now_str()

        self.output = config['output']
        self.output_runtime = os.path.join(self.output, self.runtime)

        self.shell = 'python ' + ' '.join(sys.argv)
        self.builder = builder
        self.track = config['track']

        os.makedirs(self.output_runtime, exist_ok=True)
        self.write('config.yaml', config)
        self.write('options.json', self.env.options.to_dict())
        self.write('shell.sh', self.shell)
        self.write('builder.json', self.builder.repo_desc)

        self.policies: list[Policy] = []

        for agent_name in options.agents():
            agent_color = 'red' if agent_name in options.red_agents else 'blue'
            self.add_bt_policy(
                    agent_name=agent_name,
                    filepath=config['policy'].get(agent_name, config['policy'].get(agent_color)),
                    context=config.get('context', { })
            )

    def add_bt_policy(
            self,
            agent_name: str,
            filepath: str,
            tree_name_suffix: str = '',
            context: dict = None
    ):
        tree = DogfightTree(
                env=self.env,
                agent_name=agent_name,
                root=self.builder.build_from_file(filepath),
                name=agent_name + tree_name_suffix,
                context={
                    'title'   : self.title,
                    'desc'    : self.desc,
                    'filename': self.builder.get_relative_filename(filepath=filepath),
                    'filepath': self.builder.find_filepath(filepath=filepath),
                    'output'  : self.output,
                    'runtime' : self.runtime,  # 执行时间
                    **(context or { })
                }
        ).setup()

        policy = BTPolicy(
                env=self.env,
                tree=tree,
                agent_name=agent_name,
        )

        self.write(f'{agent_name}.xml', '\n'.join([f'<!--{filepath}-->', self.bt_to_xml(tree.root)]))
        # render_node(tree.root, os.path.join(self.output_runtime, f'{agent_name}.svg'))
        board = pybts.Board(tree=policy.tree, log_dir=self.output_runtime)
        board.clear()

        def before_env_reset(env: Dogfight2dEnv):
            """保存环境的一些数据"""
            self.write(self.episode_file(f'{agent_name}.xml'), self.bt_to_xml(tree.root))
            self.write(self.episode_file(f'{agent_name}-context.json'), tree.context)
            env.update_game_info(total=True)
            board.track({
                **env.game_info,
                agent_name: env.get_agent(agent_name).to_dict(),
            })

        self.env.add_before_reset_handler(before_env_reset)

        if self.track >= 0:
            track_throttle = Throttle(duration=self.track)

            def on_tree_post_tick(t):
                if track_throttle.should_call(self.env.time):
                    self.update_game_info()
                    board.track({
                        'env_time': self.env.time,
                        **self.env.game_info,
                        # **policy.tree.context,
                        # agent_name: env.get_agent(agent_name).to_dict(),
                    })

            def on_tree_reset(t):
                track_throttle.reset()

            tree.add_post_tick_handler(on_tree_post_tick)
            tree.add_reset_handler(on_tree_reset)

        self.policies.append(policy)

    def update_game_info(self):
        color_dict = defaultdict(int)
        color_dict['red'] = 0
        color_dict['blue'] = 0
        for agent in self.env.battle_area.agents:
            for policy in self.policies:
                if isinstance(policy, BTPolicy) and policy.agent_name == agent.name:
                    rl_reward = policy.tree.context['reward']
                    for k, v in rl_reward.items():
                        color_dict[policy.agent.color] += v
                        color_dict[f'{policy.agent.color}_{k}'] += v
        for k, v in color_dict.items():
            self.env.game_info[f'reward_{k}'] = v

    def run(self, episodes: int):
        print('开始', self.config)
        env = self.env

        self.update_game_info()

        policy = MultiAgentPolicy(policies=self.policies)
        pbar = tqdm(total=episodes, desc=f'{self.title}[{self.output_runtime}]')

        start_time = time.time()
        for episode in range(episodes):
            if not env.isopen:
                break
            env.reset()

            while env.isopen:
                if env.should_update():
                    policy.take_action()
                    policy.put_action()
                    env.update()
                    info = env.gen_info()
                    if info['terminated'] or info['truncated']:
                        # 在terminated之后还要再触发一次行为树，不然没办法将最终奖励给到行为树里的节点
                        policy.take_action()
                        break

                if env.should_render():
                    env.render()

            self.update_game_info()

            self.write(self.episode_file('env_info.json'), env.gen_info())
            self.write(self.episode_file('game_info.json'), env.game_info)

            for agent in self.env.battle_area.agents:
                self.write(self.episode_file(f'{agent.name}.json'), agent.to_dict())

            cost_time = time.time() - start_time
            self.write(f'env.json', {
                'cost_time': int(cost_time),
                'info'     : env.gen_info(),
                'game_info': env.game_info,
            })

            pbar.update(1)
            pbar.set_postfix(
                    {
                        'returns' : f'{self.env.game_info["reward_red"]:.2f} vs {self.env.game_info["reward_blue"]:.2f}',
                        'wins'    : f'{self.env.game_info["red_wins"]} vs {self.env.game_info["blue_wins"]}',
                        'draws'   : f'{self.env.game_info["draws"]}',
                        'winner'  : self.env.battle_area.winner,
                        'time'    : f'{int(env.time)}/{env.battle_area.accum_time:.0f}',
                        'mis_fire': self.env.game_info["missile_fired_count"],
                        'mis_miss': self.env.game_info['missile_miss_count'],
                        'mis_hit' : f'{self.env.game_info["missile_hit_enemy_count"]}'
                        # 'collided_aircraft_count'    : self.env.game_info['collided_aircraft_count'],
                        # 'missile_fire_fail_count'    : self.env.game_info['missile_fire_fail_count']
                    })
            print()
            self.write(f'run-{episodes}.txt', f'{episode}: {pbar.postfix}\n', 'a')

        cost_time = time.time() - start_time
        self.write(f'耗时={cost_time:.0f}.txt', f'耗时: {cost_time:.2f} 秒')

    def episode_file(self, file: str):
        return os.path.join('episodes', str(self.env.game_info['episode']), file)

    def write(self, path: str, content: str | dict, mode='w') -> None:
        path = os.path.join(self.output_runtime, path)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(path, mode, encoding='utf-8') as f:
            if isinstance(content, str):
                f.write(content)
            else:
                if path.endswith('json'):
                    pybts.utility.json_dump(content, f, indent=4, ensure_ascii=False)
                else:
                    import yaml
                    yaml.safe_dump(content, f, allow_unicode=True, indent=4)

    def delete(self, path: str):
        path = os.path.join(self.output_runtime, path)
        if os.path.exists(path):
            os.remove(path)

    @classmethod
    def bt_to_xml(cls, node: pybts.behaviour.Behaviour):
        return pybts.utility.bt_to_xml(node,
                                       ignore_children=False,
                                       ignore_to_data=True,
                                       ignore_attrs=['id', 'name', 'status', 'tag', 'type', 'blackboard',
                                                     'feedback_messages',
                                                     'children_count'])


def create_manager(config: dict):
    builder = bt.CustomBTBuilder(folders=[config['output'], 'scripts'])

    manager = BTManager(
            config=config,
            builder=builder)

    return manager


if __name__ == '__main__':
    print(read_config('run.yaml')['policy'])
