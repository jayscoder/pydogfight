from __future__ import annotations

import os
from typing import Optional, Union
from PIL import Image
from minigrid.utils.rendering import *
import pygame
from typing import Tuple
from gym_dogfight.core.options import Options
import types
from gym_dogfight.core.constants import *


def pygame_load_img(path):
    cwd = os.path.dirname(__file__)
    image = pygame.image.load(os.path.join(cwd, '..', 'imgs', path))
    return image


def game_point_to_screen_point(
        game_point: Tuple[float, float],
        game_size: Tuple[float, float],
        screen_size: Tuple[float, float]) -> Tuple[
    float, float]:
    """
    Converts a game_point into a screen coordinate
    :param game_point:
    :param game_size:
    :param screen_size:
    :return:
    """
    screen_x = (game_point[0] / (
            game_size[0] / 2)) * screen_size[0] / 2 + screen_size[0] / 2
    screen_y = screen_size[1] / 2 - (
            game_point[1] / (game_size[1] / 2)) * screen_size[1] / 2
    return screen_x, screen_y


def screen_point_to_game_point(
        screen_point: Tuple[float, float],
        game_size: Tuple[float, float],
        screen_size: Tuple[float, float]) -> Tuple[float, float]:
    game_x = ((screen_point[0] - screen_size[0] / 2) * 2 / screen_size[0]) * (game_size[0] / 2)
    game_y = ((screen_size[1] / 2 - screen_point[1]) * 2 / screen_size[1]) * (game_size[1] / 2)
    return game_x, game_y


def game_length_to_screen_length(
        game_length: float,
        game_size: Tuple[float, float],
        screen_size: Tuple[float, float]
) -> float:
    max_screen_size = max(screen_size)
    max_game_size = max(game_size)
    return (game_length / max_game_size) * max_screen_size


def render_img(options: Options,
               position: tuple[float, float],
               screen,
               img_path: str,
               label: str = '',
               rotate: float = None,
               scale: Optional[tuple[float, float]] = None) -> np:
    # 加载图像（确保路径正确）
    img = pygame_load_img(img_path).convert_alpha()
    if rotate is not None:
        img = pygame.transform.rotate(img, rotate)  # 逆时针旋转
    if scale is not None:
        img = pygame.transform.smoothscale(img, scale)

    # 获取图像的矩形区域
    img_rect = img.get_rect()

    # 调整坐标系统：从游戏世界坐标转换为屏幕坐标
    screen_x, screen_y = game_point_to_screen_point(
            position,
            game_size=options.game_size,
            screen_size=options.screen_size)

    img_rect.center = (screen_x, screen_y)
    # 将飞机图像绘制到屏幕上
    screen.blit(img, img_rect)

    if label != '':
        # 创建字体对象
        font = pygame.font.Font(None, 16)  # 使用默认字体，大小为36

        # 渲染文本到 Surface 对象
        text_surface = font.render(label, True, (0, 0, 0))

        # 获取文本区域的矩形
        text_rect = text_surface.get_rect()
        text_rect.center = [img_rect.center[0], img_rect.bottom + 10]
        screen.blit(text_surface, text_rect)

    return img_rect


def render_route(options: Options, screen, route, color: str, count: int = 20):
    if route is not None:
        if isinstance(route, types.GeneratorType):
            route = list(route)
        for i in range(0, len(route), int(math.ceil(len(route) / count))):
            pygame.draw.circle(screen, COLORS[color], game_point_to_screen_point(
                    game_point=route[i][:2],
                    game_size=options.game_size,
                    screen_size=options.screen_size
            ), 1)


def render_circle(options: Options, screen, radius: float, position: tuple[float, float], color: str, width: int = 1):
    if radius <= 0:
        return
    pygame.draw.circle(screen, COLORS[color], game_point_to_screen_point(
            game_point=position,
            game_size=options.game_size,
            screen_size=options.screen_size,
    ), game_length_to_screen_length(
            radius,
            game_size=options.game_size,
            screen_size=options.screen_size
    ), width)
