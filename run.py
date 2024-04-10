import os
import sys
import utils
import yaml
import argparse
from pydogfight import Options
import time
import json

parser = argparse.ArgumentParser()
# 批量执行脚本
parser.add_argument('path', type=str, default='greedy_vs_greedy.yaml', help='配置文件路径')


def main():
    args = parser.parse_args()
    print(args.path)
    with open(args.path, 'r', encoding='utf-8') as f:
        data = list(yaml.load_all(f, Loader=yaml.FullLoader))

    config = { }
    options = Options()
    policy = { }
    for i in range(len(data)):
        config.update(data[i])
        if 'options' in config:
            options.load_dict(config['options'])
        if 'policy' in config:
            policy.update(config['policy'])

        os.makedirs(config['folder'], exist_ok=True)

        start_time = time.time()
        print(f'====== 开始执行第{i}轮配置 ======')
        print(json.dumps(config, indent=4, ensure_ascii=False))
        manager = utils.create_manager(
                folder=config['folder'],
                options=options,
                render=config.get('render', False),
                track=config.get('track', -1),
                policy_path=policy,
        )
        manager.write(path='run-config.json', content=config)
        manager.run(num_episodes=config['num_episodes'])
        cost_time = time.time() - start_time
        print(f'====== 第{i}轮配置执行结束，耗时（{cost_time:.2f} 秒） ======')


if __name__ == '__main__':
    main()
