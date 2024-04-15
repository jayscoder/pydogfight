import utils


def main():
    config = utils.read_config('run.yaml')
    manager = utils.BTManager(config, train=True)
    manager.run(episodes=config['episodes'])

if __name__ == '__main__':
    main()
