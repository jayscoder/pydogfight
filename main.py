import os
import sys
import utils
import yaml
import argparse
from pydogfight import Options
import time
import json
import threading

parser = argparse.ArgumentParser()
# 批量执行脚本
parser.add_argument('files', nargs="+", type=str, help='配置文件路径')

# 下面这些参数会覆盖配置文件里的参数
parser.add_argument('--render', action='store_true',
                    help='是否开启可视化窗口，在做强化学习训练的时候建议关闭来提高性能')
parser.add_argument('--display-tree', action='store_true',
                    help='是否将行为树的图画出来')
parser.add_argument('--train', action='store_true',
                    help='是否开启训练模式')
parser.add_argument('--output', type=str, default='', help='工作输出目录')
parser.add_argument('--episodes', type=int, default=0, help='对战场次')

args = parser.parse_args()

threads = []


def run(files: list[str]):
    global threads
    if len(files) == 1:
        run_one(files[0])
        return
    for file in files:
        if 'base' in file:
            continue
        if file.endswith('.yaml'):
            x = threading.Thread(target=run_one, args=(file,))
            threads.append(x)
            x.start()


def run_one(path: str):
    filename = os.path.basename(path).split('.')[0]
    context = {
        'filename': filename,
        'path'    : path,
        'filedir' : os.path.dirname(path)
    }
    config = utils.read_config(path, context=context)
    if args.output != '':
        config['output'] = args.output
    if args.episodes > 0:
        config['episodes'] = args.episodes
    if args.render:
        config = utils.merge_config(config, { 'options': { 'render': True } })
    if config['episodes'] > 0:
        manager = utils.BTManager(config=config, train=args.train)
        manager.run(episodes=config['episodes'])

def main():
    run(args.files)
    # 等待所有线程完成
    for index, thread in enumerate(threads):
        thread.join()


if __name__ == '__main__':
    main()
