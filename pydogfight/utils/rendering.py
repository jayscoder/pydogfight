from __future__ import annotations

from typing import Optional, Union
from typing import Tuple
from pydogfight.core.options import Options
import types
from pydogfight.core.models import BoundingBox
import math
from pydogfight.core.constants import *
import os


def pygame_load_img(path):
    import pygame
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
    w_ratio = screen_size[0] / game_size[0]
    h_ratio = screen_size[1] / game_size[1]

    screen_x = game_point[0] * w_ratio + screen_size[0] / 2
    screen_y = game_point[1] * h_ratio + screen_size[1] / 2
    return screen_x, screen_y


def screen_point_to_game_point(
        screen_point: Tuple[float, float],
        game_size: Tuple[float, float],
        screen_size: Tuple[float, float]) -> Tuple[float, float]:
    w_ratio = game_size[0] / screen_size[0]
    h_ratio = game_size[1] / screen_size[1]

    game_x = screen_point[0] * w_ratio - game_size[0] / 2
    game_y = screen_point[1] * h_ratio - game_size[1] / 2
    return game_x, game_y


def game_length_to_screen_length(
        game_length: float,
        game_size: float,
        screen_size: float,
) -> float:
    return (game_length / game_size) * screen_size


def render_img(
        options: Options,
        position: tuple[float, float],
        screen,
        img_path: str,
        label: str = '',
        rotate: float = None,
        scale: Optional[tuple[float, float]] = None) -> np:
    # 加载图像（确保路径正确）
    import pygame
    img = pygame_load_img(img_path).convert_alpha()
    if rotate is not None:
        img = pygame.transform.rotate(img, rotate)  # 顺时针旋转
    if scale is not None:
        img = pygame.transform.smoothscale(img, scale)

    # 获取图像的矩形区域
    img_rect = img.get_rect()

    # 调整坐标系统：从游戏世界坐标转换为屏幕坐标
    screen_x, screen_y = game_point_to_screen_point(
            game_point=position,
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
    import pygame
    if route is not None:
        if isinstance(route, types.GeneratorType):
            route = list(route)
        gap = max(1, int(math.ceil(len(route) / count)))
        for i in range(0, len(route), gap):
            pygame.draw.circle(screen, COLORS[color], game_point_to_screen_point(
                    game_point=route[i][:2],
                    game_size=options.game_size,
                    screen_size=options.screen_size
            ), 1)


def render_circle(options: Options, screen, radius: float, position: tuple[float, float], color: str, width: int = 1):
    if radius <= 0:
        return
    import pygame
    pygame.draw.circle(screen, COLORS[color], game_point_to_screen_point(
            game_point=position,
            game_size=options.game_size,
            screen_size=options.screen_size,
    ), game_length_to_screen_length(
            radius,
            game_size=min(options.game_size),
            screen_size=min(options.screen_size)
    ), width)


def render_rect(options: Options, screen, rect: BoundingBox, color: str, width: int = 1):
    import pygame
    left_top = game_point_to_screen_point(game_point=rect.left_top, game_size=options.game_size,
                                          screen_size=options.screen_size)
    size_w = game_length_to_screen_length(game_length=rect.size[0], game_size=options.game_size[0],
                                          screen_size=options.screen_size[0])
    size_h = game_length_to_screen_length(game_length=rect.size[1], game_size=options.game_size[1],
                                          screen_size=options.screen_size[1])
    pygame.draw.rect(screen, COLORS[color], (left_top[0], left_top[1], size_w, size_h), width=width)


def render_text(screen, text: str, topleft: tuple[float, float], text_size: int = 18, color='black'):
    import pygame
    info_font = pygame.font.Font(None, text_size)
    # 渲染文本到 Surface 对象
    text_surface = info_font.render(text, True, COLORS[color])
    # 获取文本区域的矩形
    text_rect = text_surface.get_rect()
    text_rect.topleft = topleft
    screen.blit(text_surface, text_rect)  # 绘制文本到屏幕上

# if __name__ == '__main__':
# from pydogfight.core.options import Options
# options = Options()
# p = game_point_to_screen_point(
#         game_point=options.safe_boundary.left_top,
#         game_size=options.game_size,
#         screen_size=options.screen_size
# )
# print(p)
