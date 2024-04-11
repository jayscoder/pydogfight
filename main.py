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
parser.add_argument('path', type=str, help='配置文件路径')  # 例如：run_v1.yaml

# 下面这些参数会覆盖配置文件里的参数
parser.add_argument('--render', action='store_true', help='是否开启可视化窗口，在做强化学习训练的时候建议关闭来提高性能')
parser.add_argument('--output-dir', type=str, default='', help='工作输出目录')
parser.add_argument('--track', type=int, default=-1, help='每隔几秒用pybts track一次，-1表示不track')
parser.add_argument('--train', action='store_true', help='是否开启训练模式')
parser.add_argument('--red', type=str, default='', help='红方行为树路径')
parser.add_argument('--blue', type=str, default='', help='蓝方行为树路径')
parser.add_argument('--num-episodes', type=int, default=0, help='对战场次')
parser.add_argument('--models-dir', type=str, default='', help='模型目录')
parser.add_argument('--indestructible', action='store_true',
                    help='战机是否开启无敌模式，在做强化学习训练的时候就不能靠战机是否被摧毁来获得奖励，需要靠导弹命中敌机来获得奖励')
parser.add_argument('--simulation-rate', type=int, default=0,
                    help='仿真的速率倍数，越大代表越快，update_interval内更新几次（仅在render_mode=human模式下生效）')
parser.add_argument('--record', action='store_true', help='是否记录完整的环境数据，方便之后回放')


def main():
    args = parser.parse_args()
    print(args.path)
    with open(args.path, 'r', encoding='utf-8') as f:
        data = list(yaml.load_all(f, Loader=yaml.FullLoader))

    config = {
        'output_dir'  : args.output_dir,
        'num_episodes': args.num_episodes,
        'render'      : args.render,
        'track'       : args.track,
        'options'     : {
            'train'          : args.train,
            'indestructible' : args.indestructible,
            'simulation_rate': args.simulation_rate
        },
        'context'     : {
            'models_dir': args.models_dir
        },
        'policy'      : {
            'red' : args.red,
            'blue': args.blue
        }
    }

    policy = config['policy'].copy()
    context = config['context'].copy()
    options = Options()

    for i in range(len(data)):
        config.update(data[i])

        if args.output_dir != '':
            config['output_dir'] = args.output_dir

        if args.num_episodes > 0:
            config['num_episodes'] = args.num_episodes

        if args.render:
            config['render'] = args.render

        if args.track != -1:
            config['track'] = args.track

        if args.train:
            config['options']['train'] = args.train

        if args.indestructible:
            config['options']['indestructible'] = args

        if args.simulation_rate:
            config['options']['simulation_rate'] = args.simulation_rate

        if args.models_dir != '':
            config['context']['models_dir'] = args.models_dir

        if args.red:
            config['policy']['red'] = args.red

        if args.blue:
            config['policy']['blue'] = args.blue

        if 'options' in config:
            options.load_dict(config['options'])
        if 'policy' in config:
            policy.update(config['policy'])
        if 'context' in config:
            context.update(config['context'])

        start_time = time.time()
        print(f'====== 开始执行第{i}轮配置 ======')
        print(json.dumps(config, indent=4, ensure_ascii=False))
        manager = utils.create_manager(
                output_dir=config['output_dir'],
                options=options,
                render=config['render'],
                track=config['track'],
                policy_path=policy,
                context=context
        )

        manager.write(path='run-config.json', content=config)
        manager.run(num_episodes=config['num_episodes'])
        cost_time = time.time() - start_time
        print(f'====== 第{i}轮配置执行结束，耗时（{cost_time:.2f} 秒） ======')


if __name__ == '__main__':
    main()
