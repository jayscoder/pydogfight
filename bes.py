import bezier
import numpy as np
import seaborn
from pyclothoids import SolveG2
import matplotlib.pyplot as plt


def generate_aircraft_route(
        init_pos: tuple[float, float],
        end_pos: tuple[float, float],
        init_direction: float,
        end_direction: float,
        min_turn_radius: float) -> np.ndarray:
    """
    生成无人机轨迹，并考虑最小转弯半径约束

    :param init_pos: 初始位置 (x, y)
    :param end_pos: 结束位置 (x, y)
    :param init_direction: 初始方向，以弧度为单位
    :param end_direction: 结束方向，以弧度为单位
    :param min_turn_radius: 最小转弯半径
    :return: 生成的轨迹点，形式为 (x, y) 点的数组
    """

    # Function to calculate the points of a turn given the start point, direction and radius
    def calculate_turn(center, radius, start_angle, end_angle):
        angles = np.linspace(start_angle, end_angle, num=int(radius * np.abs(end_angle - start_angle) / 0.1),
                             endpoint=True)
        return [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles]

    # Calculate the straight path direction
    straight_path_dx = end_pos[0] - init_pos[0]
    straight_path_dy = end_pos[1] - init_pos[1]
    straight_path_angle = np.arctan2(straight_path_dy, straight_path_dx)

    # Determine the turning circles for the initial and final turns
    init_turn_center = (init_pos[0] + min_turn_radius * np.sin(init_direction),
                        init_pos[1] - min_turn_radius * np.cos(init_direction))
    end_turn_center = (end_pos[0] - min_turn_radius * np.sin(end_direction),
                       end_pos[1] + min_turn_radius * np.cos(end_direction))

    # Calculate the turning points
    init_turn_points = calculate_turn(init_turn_center, min_turn_radius, init_direction, straight_path_angle)
    end_turn_points = calculate_turn(end_turn_center, min_turn_radius, straight_path_angle, end_direction)

    # Combine all the points to form the complete route
    return np.array(init_turn_points + end_turn_points)
    # # 使用 SolveG2 来生成一条满足初始和结束条件的 G2 连续路径
    # clothoid_list = SolveG2(
    #         init_pos[0], init_pos[1], init_direction, 0,  # 起点位置和初始曲率
    #         end_pos[0], end_pos[1], end_direction, 0,  # 终点位置和结束曲率
    #         Dmax=0.1, dmax=0.1  # 这些参数可以根据需要进行调整
    # )
    #
    # # 检查曲率并进行必要的调整
    # for clothoid in clothoid_list:
    #     if clothoid.KappaStart > max_curvature or clothoid.KappaEnd > max_curvature:
    #         # 需要逻辑来调整 Clothoid 曲线以满足最大曲率约束
    #         # 这可能涉及到重新计算 Clothoid 参数或修改路径
    #         print(f'clothoid KappaStart > max_curvature: {clothoid.KappaStart}')
    #         pass  # 这里需要实现具体的调整逻辑
    # # 采样生成的路径以创建连续的轨迹点
    # points = []
    # for clothoid in clothoid_list:
    #     # 假设每个Clothoid段采样足够多的点以保证平滑的曲线
    #     for s in np.linspace(0, clothoid.length, 100):
    #         x, y = clothoid.X(s), clothoid.Y(s)
    #         points.append((x, y))
    #
    # return np.array(points)


# 设置起始和结束位置、方向和最大曲率
init_pos = (0, 0)
end_pos = (10, 10)
init_direction = np.radians(45)
end_direction = np.radians(-45)
max_curvature = 0.1  # 最大曲率

# 生成曲线
curve = generate_aircraft_route(init_pos, end_pos, init_direction, end_direction, max_curvature)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(curve[:, 0], curve[:, 1], label="Clothoid Curve")
plt.scatter([init_pos[0], end_pos[0]], [init_pos[1], end_pos[1]], color='red', label="Start/End Points")
plt.quiver(init_pos[0], init_pos[1], np.cos(init_direction), np.sin(init_direction), scale=10, color='green',
           label="Initial Direction")
plt.quiver(end_pos[0], end_pos[1], np.cos(end_direction), np.sin(end_direction), scale=10, color='blue',
           label="End Direction")
plt.legend()
plt.title("Clothoid Curve Example")
plt.grid(True)
plt.show()
