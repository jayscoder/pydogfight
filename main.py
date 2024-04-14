import os
import sys
import utils
import yaml
import argparse
from pydogfight import Options
import time
import json

def main_parser():
    parser = argparse.ArgumentParser()
    # 批量执行脚本
    parser.add_argument('path', type=str, help='配置文件路径')

    # 下面这些参数会覆盖配置文件里的参数
    parser.add_argument('--render', action='store_true',
                        help='是否开启可视化窗口，在做强化学习训练的时候建议关闭来提高性能')
    parser.add_argument('--output', type=str, default='', help='工作输出目录')
    parser.add_argument('--train', action='store_true', help='是否开启训练模式')
    parser.add_argument('--red', type=str, default='', help='红方行为树路径')
    parser.add_argument('--blue', type=str, default='', help='蓝方行为树路径')
    parser.add_argument('--episodes', type=int, default=0, help='对战场次')
    parser.add_argument('--models-dir', type=str, default='', help='模型目录')
    # parser.add_argument('--record', action='store_true', help='是否记录完整的环境数据，方便之后回放')
    parser.add_argument('--rl-algo', type=str, default='', help='强化学习模型')

    args = parser.parse_args()
    config = utils.read_config(args.path)
    if args.output != '':
        config['output'] = args.output
    if args.episodes > 0:
        config['episodes'] = args.episodes
    if args.render:
        config['render'] = True
    if args.train:
        config = utils.merge_config(config, { 'options': { 'train': True } })
    if args.models_dir:
        config = utils.merge_config(config, { 'context': { 'models_dir': args.models_dir } })
    if args.red:
        config = utils.merge_config(config, { 'policy': { 'red': args.red } })
    if args.blue:
        config = utils.merge_config(config, { 'policy': { 'blue': args.blue } })
    if args.rl_algo:
        config = utils.merge_config(config, { 'context': { 'rl_algo': args.rl_algo } })

    manager = utils.create_manager(config=config)
    manager.run(config['episodes'])

if __name__ == '__main__':
    main_parser()
