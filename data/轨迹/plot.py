import matplotlib.pyplot as plt
import json


def plot_trajectories(data, title='Drone Trajectories'):
    plt.figure(figsize=(10, 6))

    for agent, trajectory in data.items():
        agent = agent.replace('_1', '')
        x_coords = [point['x'] for point in trajectory]
        y_coords = [point['y'] for point in trajectory]
        missile_fired = [point['missile_fired_count'] for point in trajectory]
        plt.plot(x_coords, y_coords, marker='.', label=agent)

        # 标记导弹发射点
        for i in range(1, len(missile_fired)):
            if missile_fired[i] > missile_fired[i - 1]:
                plt.scatter(x_coords[i], y_coords[i], color='red', s=100, edgecolors='black',
                            label=f'{agent} Missile Fired' if i == 1 else "")

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    # plt.grid(True)
    plt.savefig('D.png')


def plot_file(file, title):
    with open(file, 'r') as f:
        data = json.load(f)
    plot_trajectories(data, title=title)

    print('红方发射导弹数', data['red_1'][-1]['missile_fired_count'])
    print('蓝方发射导弹数', data['blue_1'][-1]['missile_fired_count'])


if __name__ == '__main__':
    plot_file('D.json', title='D')
