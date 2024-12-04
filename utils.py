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
from pydogfight.utils import *
from bt.nodes_rl import RLNode
from pydogfight.utils.logger import TensorboardLogger
import jinja2


def folder_run_id(folder: str):
    os.makedirs(folder, exist_ok=True)
    id_path = os.path.join(folder, "run_id.txt")
    if os.path.exists(id_path):
        with open(id_path, "r") as f:
            run_id = int(f.read())
    else:
        run_id = 0
    run_id += 1
    with open(id_path, mode="w") as f:
        f.write('{}'.format(run_id))
    return run_id


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


AGENT_INFO_KEYS = [
    'destroyed_count',
    'missile_fired_count',
    'missile_fire_fail_count',
    'missile_hit_self_count',
    'missile_hit_enemy_count',
    'missile_miss_count',
    'missile_evade_success_count',
    'home_returned_count',
    # 'missile_count',
    # 'missile_depletion_count',
    'aircraft_collided_count',
]

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


def read_config(path: str, context: dict = None):
    """读取配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        config = list(yaml.load_all(f, Loader=yaml.FullLoader))[0]

    render_config(config, context=context)

    if 'base' in config:
        config = merge_config(read_config(config['base'], context=context), config)

    if config['options'].get('title') is None and config.get('title') is not None:
        config['options']['title'] = config['title']

    if 'context' not in config['context']:
        config['context'] = { }
    config['context'] = {
        **config['context'],
        **context
    }
    return config


def render_config(config: dict, context: dict):
    for k in config:
        if isinstance(config[k], str):
            config[k] = jinja2.Template(config[k]).render(context)
        elif isinstance(config[k], dict):
            render_config(config[k], context)


def merge_config(base: dict, config: dict):
    new_config = deep_copy(base)
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

    def __init__(self, config: dict,
                 train: bool,
                 verbose: int = 0, display_tree: bool = False):
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
        self.display_tree = display_tree
        self.output = config['output']
        self.run_id = folder_run_id(self.output)
        self.shell = 'python ' + ' '.join(sys.argv)
        self.builder = bt.CustomBTBuilder(folders=[config['output'], 'scripts'])
        self.track = config['track']
        self.output_run_id = os.path.join(self.output, str(self.run_id))
        os.makedirs(self.output_run_id, exist_ok=True)

        self.write('config.yaml', config)
        self.write('options.json', self.env.options.to_dict())
        self.write('shell.sh', self.shell)
        self.write('builder.yaml', self.builder.repo_desc)

        self.policies: list[Policy] = []

        self.board_dict = { }
        for agent_name in options.agents():
            agent_color = 'red' if agent_name in options.red_agents else 'blue'
            self.add_bt_policy(
                    agent_name=agent_name,
                    filepath=config['policy'].get(agent_name, config['policy'].get(agent_color)),
                    context=config.get('context', { })
            )

        self.result_recorder = ResultRecorder(env=self.env, policies=self.policies)
        self.move_saver = ModelSaver(
                models_dir=os.path.join(self.output, 'models'),
                env=self.env,
                policies=self.policies)

        self.env.add_episode_start_handler(lambda _: self.on_episode_start())
        self.env.add_episode_end_handler(lambda _: self.on_episode_end())
        self.env.add_after_update_handler(lambda _: self.after_update())

        self.train = train
        self.pbar: tqdm | None = None

        self.logger_dict: dict[str, TensorboardLogger] = { }
        for color in ['red', 'blue']:
            self.logger_dict[color] = TensorboardLogger(os.path.join(self.output_run_id, color), verbose=verbose)

        self.start_time = time.time()

    def update_reward_to_game_info(self, reward_dict: dict):
        new_game_info = {
            'red'  : { },
            'blue' : { },
            'agent': { }
        }
        for agent_name in reward_dict:
            if agent_name not in new_game_info['agent']:
                new_game_info['agent'][agent_name] = { }
            agent = self.env.get_agent(agent_name)
            for k, v in reward_dict[agent_name].items():
                dict_incr(new_game_info[agent.color], 'reward', v)
                dict_incr(new_game_info[agent.color], f'reward_{k}', v)

                dict_incr(new_game_info['agent'][agent.name], 'reward', v)
                dict_incr(new_game_info['agent'][agent.name], f'reward_{k}', v)

        # for color in ['red', 'blue']:
        #     enemy_color = calc_enemy_color(color)
        #     if 'reward' in new_game_info[color] and 'reward' in new_game_info[enemy_color]:
        #         is_win = new_game_info[color]['reward'] > new_game_info[enemy_color]['reward']
        #         is_lose = new_game_info[color]['reward'] < new_game_info[enemy_color]['reward']
        #         is_draw = new_game_info[color]['reward'] == new_game_info[enemy_color]['reward']
        #
        #         # 这几个是累积量
        #         dict_incr(self.env.game_info[color], 'reward_win', int(is_win))
        #         dict_incr(self.env.game_info[color], 'reward_lose', int(is_lose))
        #         dict_incr(self.env.game_info[color], 'reward_draw', int(is_draw))
        #
        #         for k in ['win', 'lose', 'draw']:
        #             new_game_info[color][f'reward_{k}_rate'] = self.env.game_info[color][
        #                                                            f'reward_{k}'] / self.env.episode
        #
        merge_tow_dicts(new_game_info, self.env.game_info)

    def on_episode_end(self):
        reward_dict = { }
        for policy in self.policies:
            if not isinstance(policy, BTPolicy):
                continue
            if self.track > 0:
                board = self.board_dict[policy.agent_name]
                board.track({
                    **self.env.game_info,
                    policy.agent_name: self.env.get_agent(policy.agent_name).to_dict(),
                })

            reward_dict[policy.agent.name] = deep_copy(policy.tree.context.get('reward', { }))
        self.update_reward_to_game_info(reward_dict=reward_dict)
        self.env.game_info['recent'] = self.result_recorder.record()

        # if self.track > 0:
        #     self.write(self.episode_file('env_info.json'), self.env.gen_info())
        #     self.write(self.episode_file('game_info.json'), self.env.game_info)

        # for agent in self.env.battle_area.agents:
        #     self.write(self.episode_file(f'{agent.name}.json'), agent.to_dict())

        # self.write(f'env.json', {
        #     'info'     : self.env.gen_info(),
        #     'game_info': self.env.game_info,
        # })

        if self.train and self.env.episode > 0 and self.env.episode % self.result_recorder.recent == 0:
            self.move_saver.check_save()

        self.pbar.set_postfix(
                {
                    'reward': f"{dict_get(self.env.game_info, 'red.reward', 0):.2f} vs {dict_get(self.env.game_info, 'blue.reward', 0):.2f}",
                    # 'reward_evade': f"{dict_get(self.env.game_info, 'red.reward_evade', 0):.2f} vs {dict_get(self.env.game_info, 'blue.reward_evade', 0):.2f}",
                    'winner': self.env.battle_area.winner,
                    'r_win' : f"{dict_get(self.env.game_info, 'recent.red.win', 0)}:{dict_get(self.env.game_info, 'recent.blue.win', 0)}:{dict_get(self.env.game_info, 'recent.red.draw', 0)}",
                    'win'   : f"{self.env.game_info['red']['win']}:{self.env.game_info['blue']['win']}:{self.env.game_info['red']['draw']}",
                    # 'draw'      : f"{self.env.game_info['red']['draw']}",
                    'time'  : f'{self.env.time:.0f}/{self.env.battle_area.accum_time:.0f}',
                    # 'm_fire'    : f"{self.env.game_info['red']['missile_fired_count']} vs {self.env.game_info['blue']['missile_fired_count']}",
                    # 'm_miss'    : f"{self.env.game_info['red']['missile_miss_count']} vs {self.env.game_info['blue']['missile_miss_count']}",
                    # 'm_hit'     : f"{self.env.game_info['red']['missile_hit_enemy_count']} vs {self.env.game_info['blue']['missile_hit_enemy_count']}"
                    # 'collided_aircraft_count'    : self.env.game_info['collided_aircraft_count'],
                    # 'missile_fire_fail_count'    : self.env.game_info['missile_fire_fail_count']
                })

        for color in ['red', 'blue']:
            self.logger_dict[color].record(f'env/time', self.env.time)
            self.logger_dict[color].record(f'env/cost_time', time.time() - self.start_time)

        for agent in self.env.battle_area.agents:
            # 存活时间
            self.logger_dict[agent.color].record_mean(f'agent/survival_time', agent.survival_time)
            # 平均存活时间
            self.logger_dict[agent.color].record_mean_weighted(f'agent/survival_time', agent.survival_time)
            # # 规避成功率
            self.logger_dict[agent.color].record_mean_weighted(f'agent/missile_evade_success_rate',
                                                               agent.missile_evade_success_count,
                                                               agent.missile_evade_success_count + agent.missile_hit_self_count)

            self.logger_dict[agent.color].record_mean_weighted(f'agent/missile_evade_success_rate',
                                                               agent.missile_evade_success_count,
                                                               agent.missile_evade_success_count + agent.missile_hit_self_count)

            # 命中率
            self.logger_dict[agent.color].record_mean_weighted(f'agent/missile_hit_enemy_rate',
                                                               agent.missile_hit_enemy_count,
                                                               agent.missile_fired_count)
            self.logger_dict[agent.color].record_mean_weighted(f'agent/missile_hit_enemy_rate',
                                                               agent.missile_hit_enemy_count,
                                                               agent.missile_fired_count)

        for color in ['red', 'blue']:

            for k in ['win', 'lose', 'draw', 'win_rate', 'draw_rate', 'lose_rate']:
                self.logger_dict[color].record(f'env/{k}', self.env.game_info[color][k])
                recent_v = dict_get(self.env.game_info, ['recent', color, k])
                if recent_v is not None:
                    self.logger_dict[color].record(f'recent/{k}', recent_v)

            for k in self.env.game_info[color]:
                if k.startswith('reward'):
                    self.logger_dict[color].record(f'reward/{k}', self.env.game_info[color][k])
                    # recent_v = dict_get(self.env.game_info, ['recent', color, k])
                    # if recent_v is not None:
                    #     self.logger_dict[color].record(f'recent/{k}', recent_v)

            # for k in AGENT_INFO_KEYS:
            #     if k not in self.env.game_info[color]:
            #         continue
            #     self.logger_dict[color].record_mean_weighted(f'agent/{k}', self.env.game_info[color][k])
            # recent_v = dict_get(self.env.game_info, ['recent', color, k])
            # if recent_v is not None:
            #     self.logger_dict[color].record_mean(f'recent/{k}', recent_v)

        print()
        self.pbar.update(1)
        for k, v in self.logger_dict.items():
            v.dump(self.env.episode)
        self.write(f'run.txt', f'{self.env.episode}: {self.pbar.postfix}\n', 'a')

    def on_episode_start(self):
        pass

    def after_update(self):
        self.update_render_info()

    def add_bt_policy(
            self,
            agent_name: str,
            filepath: str,
            tree_name_suffix: str = '',
            context: dict = None
    ):
        self.builder.context = context
        tree = DogfightTree(
                env=self.env,
                agent_name=agent_name,
                root=self.builder.build_from_file(filepath),
                name=agent_name + tree_name_suffix,
                context={
                    'title'        : self.title,
                    'desc'         : self.desc,
                    'filename'     : self.builder.get_relative_filename(filepath=filepath),
                    'filepath'     : self.builder.find_filepath(filepath=filepath),
                    'output'       : self.output,
                    'output_run_id': self.output_run_id,
                    'run_id'       : self.run_id,
                    'episode'      : self.env.episode,
                    'agent_name'   : agent_name,
                    **(context or { })
                }
        ).setup()

        policy = BTPolicy(
                env=self.env,
                tree=tree,
                agent_name=agent_name,
        )

        board = pybts.Board(tree=policy.tree, log_dir=self.output_run_id)
        board.clear()
        self.board_dict[agent_name] = board

        self.write(f'{agent_name}.xml', '\n'.join([f'<!--{filepath}-->', self.bt_to_xml(tree.root)]))

        if self.display_tree:
            render_node(tree.root, os.path.join(self.output_run_id, f'{agent_name}.png'))

        if self.track >= 0:
            track_throttle = Throttle(duration=self.track)

            def on_tree_post_tick(t):
                if track_throttle.should_call(self.env.time):
                    board.track({
                        **self.env.game_info,
                        'context': policy.tree.context,
                        # agent_name: env.get_agent(agent_name).to_dict(),
                    })

            def on_tree_reset(t):
                track_throttle.reset()

            tree.add_post_tick_handler(on_tree_post_tick)
            tree.add_reset_handler(on_tree_reset)

        self.policies.append(policy)

    def update_render_info(self):
        render_info = []
        for key in [
            'episode', 'time', 'accum_time',
        ]:
            render_info.append(f'{key}: {self.env.game_info[key]}')

        for k in ['win', 'win_rate', 'draw', 'reward']:
            if k not in self.env.game_info['red'] or k not in self.env.game_info['blue']:
                continue
            red_v = round(self.env.game_info['red'][k], 2)
            blue_v = round(self.env.game_info['blue'][k], 2)
            render_info.append(
                    f"{k}: {red_v} vs {blue_v}")

        if 'recent' in self.env.game_info:
            for k in ['win', 'win_rate', 'reward']:
                red_v = round(dict_get(self.env.game_info, f'recent.red.{k}', 0), 2)
                blue_v = round(dict_get(self.env.game_info, f'recent.blue.{k}', 0), 2)
                render_info.append(
                        f"recent_{k}: {red_v} vs {blue_v}")

        for key in AGENT_INFO_KEYS:
            if key not in self.env.game_info['red'] or key not in self.env.game_info['blue']:
                continue
            red_count = self.env.game_info['red'][key]
            blue_count = self.env.game_info['blue'][key]
            render_info.append(f'{key}: {red_count} vs {blue_count}')

        self.env.render_info = render_info

    def update_context(self, context: dict):
        for policy in self.policies:
            if isinstance(policy, BTPolicy):
                policy.tree.context.update(context)

    def run(self, episodes: int):
        self.pbar = tqdm(total=episodes, desc=f'[{self.output_run_id}] train={self.train}')
        self.update_context({
            'train'  : self.train,
            'episode': self.env.episode,
        })

        print('开始', self.config)
        env = self.env

        policy = MultiAgentPolicy(policies=self.policies)

        self.start_time = time.time()

        env.reset()
        env_time = 0
        recorder: TrajRecorder | None = None

        if not self.train:
            pass
            # recorder = TrajRecorder(env=self.env, output_dir=self.output)

        for i in range(episodes):
            if not env.isopen:
                break
            self.update_context({
                'episode': self.env.episode,
            })
            while env.isopen:
                if env.should_update():
                    policy.take_action()
                    policy.put_action()
                    env.update()
                    if recorder is not None:
                        recorder.record()
                    info = env.gen_info()
                    if info['terminated'] or info['truncated']:
                        policy.take_action()
                        # 在terminated之后还要再触发一次行为树，不然没办法将最终奖励给到行为树里的节点
                        # 而且需要强制将所有的RLNode都触发一遍，避免因为条件节点关系部分漏掉
                        for p in self.policies:
                            if isinstance(p, BTPolicy):
                                for node in p.tree.root.iterate():
                                    if isinstance(node, RLNode):
                                        node.take_action()
                        break

                if env.should_render():
                    env.render()

            if recorder is not None:
                recorder.save()
            env_time += self.env.time
            env.reset()
            policy.reset()

        cost_time = time.time() - self.start_time
        self.write(f'耗时={cost_time:.0f}.txt', '\n'.join(
                [
                    f'耗时: {cost_time:.2f} 秒',
                    f'平均耗时 {cost_time / episodes: .2f}秒',
                    f'平局局时 {env_time / episodes: .0f}秒'
                ]
        ))

    def episode_file(self, file: str):
        return os.path.join('episodes', str(self.env.episode), file)

    def write(self, path: str, content: str | dict, mode='w') -> None:
        path = os.path.join(self.output_run_id, path)
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
        path = os.path.join(self.output_run_id, path)
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


class TrajRecorder:
    """
    对战轨迹记录
    """

    def __init__(self, env: Dogfight2dEnv, output_dir: str):
        self.output_dir = os.path.join(output_dir, 'record')
        os.makedirs(self.output_dir, exist_ok=True)
        self.env = env
        self.data = { }

    def record(self):
        for agent in self.env.battle_area.agents:
            if agent.name not in self.data:
                self.data[agent.name] = []
            self.data[agent.name].append({
                'x'                          : float(agent.waypoint.x),
                'y'                          : float(agent.waypoint.y),
                'psi'                        : float(agent.waypoint.psi),
                'missile_fired_count'        : int(agent.missile_fired_count),
                'missile_miss_count'         : int(agent.missile_miss_count),
                'missile_evade_success_count': int(agent.missile_evade_success_count)
            })

    def save(self):
        with open(os.path.join(self.output_dir, f'{self.env.episode}.json'), 'w') as f:
            json.dump(self.data, f, ensure_ascii=False)
        self.data = { }


class ResultRecorder:
    """
    对战结果记录
    """

    def __init__(self, env: Dogfight2dEnv, policies: list[Policy], recent: int = 50):
        self.env = env
        self.policies = policies
        self.game_infos = []
        self.agent_infos = []
        self.recent = recent

    def record(self):
        # 记录对战结果到game_info中
        self.game_infos.append(deep_copy(self.env.game_info))
        if len(self.game_infos) > self.recent + 1:
            self.game_infos = self.game_infos[-(self.recent + 1):]

        return self.compute_recent_stats()

    def compute_recent_stats(self) -> dict:
        """计算最近N场对局的信息"""
        if len(self.game_infos) == 0:
            return { }

        end_game_info = self.game_infos[-1]
        if len(self.game_infos) > self.recent:
            episodes = len(self.game_infos) - 1
            start_game_info = self.game_infos[0]
        else:
            episodes = len(self.game_infos)
            start_game_info = { }

        stats = {
            'episodes': episodes,
            'abnormal': [],
            'red'     : { },
            'blue'    : { },
            'agent'   : { }
        }

        # 需要减去初始值的量
        for color in ['red', 'blue']:
            for k in ['win', 'lose', 'draw']:
                stats[color][k] = end_game_info[color][k] - dict_get(start_game_info, [color, k], 0)

            for k in end_game_info[color]:
                if k.startswith('reward') and not k.endswith('rate'):
                    stats[color][k] = end_game_info[color][k] - dict_get(start_game_info, [color, k], 0)

        for color in ['red', 'blue']:
            for k in ['win', 'lose', 'draw', 'reward_win', 'reward_lose', 'reward_draw']:
                stats[color][f'{k}_rate'] = dict_get(stats, [color, k], 0) / episodes

        for color in ['red', 'blue']:
            enemy_color = 'blue' if color == 'red' else 'red'
            if stats[color]['win_rate'] > stats[enemy_color]['win_rate']:
                if stats[color]['reward_win_rate'] < stats[enemy_color]['reward_win_rate']:
                    # 奖励和胜率不匹配
                    stats['abnormal'].append(color)

        # if len(stats['abnormal']) > 0:
        #     print(
        #             f"[{self.env.episode}]最近{stats['episodes']}轮对战出现奖励和胜率倒挂异常: {stats['abnormal']}",
        #             stats)

        # agent的数据需要累积平均
        for item in self.game_infos:
            for k in AGENT_INFO_KEYS:
                for color in ['red', 'blue']:
                    dict_incr(stats[color], k, dict_get(item, [color, k], 0))
            for agent_name in item['agent']:
                if 'agent_name' not in stats['agent']:
                    stats['agent'][agent_name] = { }
                for k in AGENT_INFO_KEYS:
                    dict_incr(stats['agent'][agent_name], k, dict_get(item, ['agent', agent_name, k], 0))

        for k in AGENT_INFO_KEYS:
            for color in ['red', 'blue']:
                stats[color][k] /= episodes

        for agent_name in stats['agent']:
            for k in AGENT_INFO_KEYS:
                stats['agent'][agent_name][k] /= episodes

        return stats


class ModelSaver:

    def __init__(self, models_dir: str, env: Dogfight2dEnv, policies: list[Policy]):
        self.models_dir = models_dir
        self.env = env
        self.policies = policies
        os.makedirs(self.models_dir, exist_ok=True)

    def check_save(self):
        # 检查是否需要保存模型，只保存最近胜率最高的模型
        for policy in self.policies:
            if not isinstance(policy, BTPolicy):
                continue
            for node in policy.tree.root.iterate():
                if not isinstance(node, RLNode):
                    continue
                if self.should_save_model(node):
                    self.save_model(node)

    def should_save_model(self, node: RLNode) -> bool:
        """如果新的更好，则返回true，否则返回false"""
        node_path = os.path.join(self.models_dir, f'{node.name}.json')
        if not os.path.exists(node_path):
            return True
        with open(node_path, 'r', encoding='utf-8') as file:
            old_data = json.load(file)

        new_color = node.agent.color
        old_color = old_data['agent']['color']
        # 最近胜率高的好
        if self.env.game_info['recent'][new_color]['win_rate'] > old_data['game_info']['recent'][old_color]['win_rate']:
            return True
        return False

    def save_model(self, node: RLNode) -> None:
        node.save_model(os.path.join(self.models_dir, node.name))

        with open(os.path.join(self.models_dir, f'{node.name}.json'), 'w',
                  encoding='utf-8') as file:
            pybts.utility.json_dump({
                **self.env.game_info['recent'][node.agent.color],
                'agent'    : node.agent.to_dict(),
                'game_info': self.env.game_info,
            }, file, indent=4, ensure_ascii=False)
