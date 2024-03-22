import numpy as np
import os
from matplotlib import pyplot as plt
import math
from gym_dogfight.algos.geometry import *


def plot_tangent_circles(x0, y0, angle, radius):
    # 计算圆心
    centers = calculate_tangent_circle_centers(x0, y0, angle, radius)

    # 创建图和坐标轴
    fig, ax = plt.subplots()

    # 画出两个圆
    circle1 = plt.Circle(centers[0], radius, color='blue', fill=False)
    circle2 = plt.Circle(centers[1], radius, color='red', fill=False)
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    # 画出切线
    line_length = 100  # 直线的长度
    theta = math.radians(angle)
    x1 = x0 + line_length * math.cos(theta)
    y1 = y0 + line_length * math.sin(theta)
    plt.plot([x0, x1], [y0, y1], 'green')

    # 画出切点
    plt.scatter([x0], [y0], color='black')

    # 设置图形的显示范围
    plt.xlim(x0 - radius * 2, x0 + radius * 2)
    plt.ylim(y0 - radius * 2, y0 + radius * 2)

    # 设置坐标轴比例相同，并显示图形
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()


def plot_circle_tangent_points(x0, y0, r, x1, y1):
    points = calculate_circle_tangent_points(x0, y0, r, x1, y1)

    # 创建图和坐标轴
    fig, ax = plt.subplots()

    # 画出圆
    circle1 = plt.Circle((x0, y0), r, color='blue', fill=False)
    ax.add_artist(circle1)

    for point in points:
        # 画出切线
        plt.plot([point[0], x1], [point[1], y1], 'green')
        # 画出切点
        # plt.scatter([point[0]], [point[1]], color='black')

    plot_size = max(x0, x1, y0, y1) + r * 2
    # 设置图形的显示范围
    plt.xlim(-plot_size, plot_size)
    plt.ylim(-plot_size, plot_size)

    # 设置坐标轴比例相同，并显示图形
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'outputs/{x0}_{y0}_{r}_{x1}_{y1}.png')


def test_calculate_tangent_circle_centers():
    plot_tangent_circles(0, 0, 45, 5)


def test_calculate_circle_tangent_lines():
    for x in range(-10, 10, 2):
        for y in range(-10, 10, 2):
            plot_circle_tangent_points(0, 0, 5, x, y)
