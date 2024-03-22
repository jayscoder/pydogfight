from __future__ import annotations

import os
from PIL import Image
from minigrid.utils.rendering import *
import pygame
from typing import Tuple


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
