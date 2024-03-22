# from numpy import pi
# import matplotlib.pyplot as plt
# from pyclothoids import Clothoid
# clothoid0 = Clothoid.G1Hermite(0, 0, pi, 1, 1, 0)
# plt.plot( *clothoid0.SampleXY(500) )
# print(clothoid0.dk, clothoid0.KappaStart, clothoid0.KappaEnd)
# Use the parameter env_id to make the environment
# env = gym.make(env_id, render_mode='human')
import time

from gym_dogfight import *

# env = gym.make('Dogfight-v0', render_mode="human")  # for visual debugging
env = Dogfight2dEnv(render_mode='human')
if __name__ == '__main__':
    env.reset()

    for i in range(100):
        env.step(None)
        print(env.battle_area.duration)

# import pygame
# import sys
#
# # 初始化 Pygame
# pygame.init()
#
# # 设置窗口大小
# width, height = 640, 480
# screen = pygame.display.set_mode((width, height))
#
# # 设置窗口标题
# pygame.display.set_caption('Pygame Demo')
#
# # 定义颜色
# black = (255, 255, 255)
# red = (255, 0, 0)
#
# # 矩形初始位置和大小
# rect_x, rect_y = 50, 50
# rect_width, rect_height = 60, 60
#
# # 移动速度
# velocity = 5
#
# # 游戏主循环
# running = True
# while running:
#     # 处理事件
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#
#     # 清屏
#     screen.fill(black)
#
#     # 绘制矩形
#     pygame.draw.rect(screen, red, (rect_x, rect_y, rect_width, rect_height))
#
#     # 更新矩形位置
#     rect_x += velocity
#     if rect_x + rect_width > width or rect_x < 0:
#         velocity = -velocity
#
#     # 更新屏幕显示
#     pygame.display.flip()
#
#     # 控制帧率
#     pygame.time.Clock().tick(60)
#
# # 退出 Pygame
# pygame.quit()
# sys.exit()
