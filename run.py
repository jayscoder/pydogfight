import utils
import os


def main(path: str):
    filename = os.path.basename(path).split('.')[0]
    context = {
        'filename': filename,
        'path'    : path,
        'filedir' : os.path.dirname(path)
    }
    config = utils.read_config(path, context=context)
    manager = utils.BTManager(config, train=True)
    manager.run(episodes=config['episodes'])


if __name__ == '__main__':
    main(path='scripts/thesis/ppo-A_vs_greedy.yaml')
