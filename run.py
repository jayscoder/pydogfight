import utils


def main():
    config = utils.read_config('run.yaml')
    manager = utils.create_manager(config=config)
    manager.run(config['episodes'])


if __name__ == '__main__':
    main()
