from __future__ import annotations
from pydogfight.policy.bt.base_class import *
from pydogfight.core.actions import Actions
import pybts
from pybts import Status
import os


class ManualControl(BTPolicyNode):
    def update(self) -> Status:
        import pygame
        event = pygame.event.poll()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # 朝着最近的敌人发射导弹
                enemy = self.env.battle_area.find_nearest_enemy(self.agent_name)
                if enemy is not None:
                    self.actions.put_nowait([Actions.fire_missile, enemy.waypoint.x, enemy.waypoint.y])
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 获取鼠标点击的位置
            from pydogfight.utils.rendering import screen_point_to_game_point
            position = screen_point_to_game_point(
                    screen_point=pygame.mouse.get_pos(),
                    game_size=self.env.options.game_size,
                    screen_size=self.env.options.screen_size
            )

            self.actions.put_nowait([Actions.go_to_location, position[0], position[1]])

        return Status.SUCCESS
