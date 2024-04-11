from pydogfight import Options

if __name__ == '__main__':
    for i in range(-1, 2):
        for j in range(-1, 2):
            print(Options.game_size[0] * i * 0.7, Options.game_size[1] * j)
