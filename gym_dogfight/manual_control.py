#!/usr/bin/env python3

from __future__ import annotations

import time

import gymnasium as gym
import pygame
from gymnasium import Env

from gym_dogfight.core.actions import Actions
from gym_dogfight.envs.dogfight_2d_env import Dogfight2dEnv
from gym_dogfight.core.world_obj import Aircraft


class ManualControl:
    def __init__(
            self,
            env: Dogfight2dEnv,
            control_agents: list[str] | None = None,
            seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.control_agents = control_agents or self.env.options.agents  # 可以操控的agents

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        current_agent_index = 0
        while not self.closed:
            agent_name = self.control_agents[current_agent_index]
            self.env.render_info['duration'] = f'{self.env.duration:.0f} s'
            self.env.render_info['agent'] = agent_name
            agent_obj = self.env.get_agent(agent_name)
            self.env.render_info['x'] = f'{agent_obj.x:.0f}'
            self.env.render_info['y'] = f'{agent_obj.y:.0f}'
            self.env.render_info['psi'] = f'{agent_obj.psi:.0f}'
            self.env.render_info['fuel'] = f'{agent_obj.fuel:.0f}'
            self.env.render_info['missile'] = f'{agent_obj.missile_count}'
            self.env.render_info['destroy_enemy'] = f'{agent_obj.missile_destroyed_agents}'

            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                self.env.close()
                break
            if event.type == pygame.KEYDOWN:
                event_key = pygame.key.name(int(event.key))
                if event_key == "escape":
                    self.env.close()
                    return
                if event_key == "backspace":
                    self.reset()
                    return
                if event.key == pygame.K_TAB:
                    current_agent_index = (current_agent_index + 1) % len(self.control_agents)
                elif event.key == pygame.K_SPACE:
                    # 朝着最近的敌人发射导弹
                    enemy = self.env.battle_area.find_nearest_enemy(agent_name)
                    if enemy is not None:
                        actions = self.env.empty_action()
                        actions[current_agent_index, 0] = Actions.fire_missile
                        actions[current_agent_index, 1] = enemy.x
                        actions[current_agent_index, 2] = enemy.y
                        self.step(actions)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 获取鼠标点击的位置
                from gym_dogfight.utils.rendering import screen_point_to_game_point
                position = screen_point_to_game_point(
                        screen_point=pygame.mouse.get_pos(),
                        game_size=self.env.options.game_size,
                        screen_size=self.env.options.screen_size
                )
                actions = self.env.empty_action()
                actions[current_agent_index, 0] = Actions.go_to_location
                actions[current_agent_index, 1] = position[0]
                actions[current_agent_index, 2] = position[1]
                self.step(actions)

            if self.env.should_update():
                self.env.update()

            if self.env.should_render():
                enemy = self.env.battle_area.find_nearest_enemy(agent_name)
                if enemy is not None:
                    self.env.render_info['nearest enemy'] = enemy.name
                    self.env.render_info['nearest enemy distance'] = f'{agent_obj.distance(enemy):.0f}'
                    hit_point = agent_obj.predict_missile_intercept_point(enemy=enemy)
                    if hit_point is not None:
                        param = agent_obj.calc_optimal_path(target=enemy.location, turn_radius=agent_obj.turn_radius)
                        self.env.render_info['nearest enemy predict hit distance'] = f'{param.length:.0f}'
                    else:
                        self.env.render_info['nearest enemy predict hit distance'] = f'inf'
                else:
                    self.env.render_info['nearest enemy'] = ''
                    self.env.render_info['nearest enemy distance'] = ''
                    self.env.render_info['missile hit prob'] = 0

                _, reward, terminated, truncated, info = self.env.step(self.env.empty_action())
                self.env.render()
                if terminated or truncated:
                    time.sleep(1)
                    self.reset(self.seed)


    def step(self, action: Actions):
        _, reward, terminated, truncated, info = self.env.step(action)

    def reset(self, seed=None):
        self.env.reset(seed=seed)

# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#             "--env-id",
#             type=str,
#             help="gym environment to load",
#             choices=gym.envs.registry.keys(),
#             default="MiniGrid-MultiRoom-N6-v0",
#     )
#     parser.add_argument(
#             "--seed",
#             type=int,
#             help="random seed to generate the environment with",
#             default=None,
#     )
#     parser.add_argument(
#             "--tile-size", type=int, help="size at which to render tiles", default=32
#     )
#     parser.add_argument(
#             "--agent-view",
#             action="store_true",
#             help="draw the agent sees (partially observable view)",
#     )
#     parser.add_argument(
#             "--agent-view-size",
#             type=int,
#             default=7,
#             help="set the number of grid spaces visible in agent-view ",
#     )
#     parser.add_argument(
#             "--screen-size",
#             type=int,
#             default="640",
#             help="set the resolution for pygame rendering (width and height)",
#     )
#
#     args = parser.parse_args()
#
#     env: MiniGridEnv = gym.make(
#             args.env_id,
#             tile_size=args.tile_size,
#             render_mode="human",
#             agent_pov=args.agent_view,
#             agent_view_size=args.agent_view_size,
#             screen_size=args.screen_size,
#     )
#
#     # TODO: check if this can be removed
#     if args.agent_view:
#         print("Using agent view")
#         env = RGBImgPartialObsWrapper(env, args.tile_size)
#         env = ImgObsWrapper(env)
#
#     manual_control = ManualControl(env, seed=args.seed)
#     manual_control.start()
