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
    parser.add_argument('--episodes', type=int, default=0, help='对战场次')

    args = parser.parse_args()
    config = utils.read_config(args.path)
    if args.output != '':
        config['output'] = args.output
    if args.episodes > 0:
        config['episodes'] = args.episodes
    if args.render:
        config = utils.merge_config(config, { 'options': { 'render': True } })

    manager = utils.BTManager(config=config, train=False)
    manager.run(episodes=config['episodes'])


if __name__ == '__main__':
    main_parser()
