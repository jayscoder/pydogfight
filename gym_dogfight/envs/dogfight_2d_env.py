from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import pygame.time
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.error import DependencyNotInstalled
from gym_dogfight.core.world_obj import *
from gym_dogfight.core.options import Options
from gym_dogfight.core.battle_area import BattleArea


class Dogfight2dEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        "render_fps"  : 50,
    }

    def __init__(self,
                 options: Options = Options(),
                 render_mode: str | None = None):
        super().__init__()
        assert len(options.red_agents) > 0 and len(options.blue_agents) > 0
        self.render_mode = render_mode
        self.options = options

        # (action_type, x, y)
        self.action_space = gym.spaces.Box(
                low=-options.game_size[0] / 2,
                high=options.game_size[1] / 2,
                shape=(len(options.red_agents) + len(options.blue_agents), 3),
                dtype=np.float32)

        # color, destroyed, x, y, psi, speed, missile_count, fuel, radar_radius
        self.observation_space = gym.spaces.Box(
                low=-options.game_size[0] / 2,
                high=options.game_size[1] / 2,
                shape=(len(options.red_agents) + len(options.blue_agents), 9),
                dtype=np.float32)

        self.screen = None
        self.clock: pygame.time.Clock | None = None
        self.isopen = True

        self.battle_area = BattleArea(options=options, render_mode=self.render_mode)

        self.last_render_time = 0
        self.step_count = 0

        self.render_info = { }

    @property
    def duration(self):
        return self.battle_area.duration

    def get_agent(self, name) -> Aircraft:
        obj = self.battle_area.get_obj(name)
        assert isinstance(obj, Aircraft)
        return obj

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.battle_area.reset()

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode(self.options.screen_size,
                                                      pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.screen.fill((255, 255, 255))
                pygame.display.set_caption("dogfight")
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.render()
        return self.gen_obs(), { }

    def gen_obs(self):

        obs = np.zeros(self.observation_space.shape)
        i = 0
        for red_name in self.options.red_agents:
            obj = self.battle_area.get_obj(red_name)
            assert isinstance(obj, Aircraft)
            obs[i, :] = [
                COLOR_TO_IDX[obj.color],
                float(obj.destroyed),
                obj.x,
                obj.y,
                obj.psi,
                obj.speed,
                obj.missile_count,
                obj.fuel,
                obj.radar_radius
            ]

        return obs

    def gen_info(self) -> dict:
        info = {
            'truncated' : self.duration > self.options.max_duration,
            'terminated': False
        }
        red_count, blue_count = self.battle_area.remain_count

        if red_count == 0 or blue_count == 0:
            if red_count > 0:
                info['winner'] = 'red'
            elif blue_count > 0:
                info['winner'] = 'blue'
            else:
                info['winner'] = 'draw'
            info['terminated'] = True
        else:
            info['winner'] = ''
        return info

    def empty_action(self):
        return np.zeros(self.action_space.shape)

    def put_action(self, action):
        for i, act in enumerate(action):
            if act[0] == 0:
                continue
            agent_name = self.options.agents[i]
            agent = self.battle_area.get_obj(agent_name)
            agent.put_action(act)

    def update(self):
        # if delta_time is None:
        #     if self.render_mode == 'human':
        #         delta_time = (pygame.time.get_ticks() - self.last_render_time) * self.options.simulation_rate / 1000
        #     else:
        #         delta_time = 1
        # # if delta_time is None:
        # #     delta_time = (self.options.render_speed / self.metadata['render_fps'])
        self.battle_area.update(delta_time=self.options.delta_time * self.options.simulation_rate)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self.put_action(action)

        # 检查是否所有的实体都被摧毁了
        terminated = False
        truncated = False
        reward = self.options.step_punish_reward
        info = self.gen_info()

        if info['winner'] != '':
            # 某一方获胜了
            terminated = True
            if self.options.self_side == info['winner']:
                reward = self.options.win_reward
            elif info['winner'] == 'draw':
                reward = self.options.draw_reward
            else:
                reward = self.options.lose_reward

        if self.battle_area.duration > self.options.max_duration:
            truncated = True

        if self.render_mode == 'human':
            if 'reward' not in self.render_info:
                self.render_info['reward'] = 0
            self.render_info['reward'] += reward

        # 注意这里返回的状态是更新动作之前的（想获取更新动作之后的状态需要手动调用update）
        return self.gen_obs(), reward, terminated, truncated, self.gen_info()

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                    "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        # print(f'Render {self.duration} {self.last_render_time}')
        self.last_render_time = pygame.time.get_ticks()

        if self.render_mode == "human":
            self.screen.fill((255, 255, 255))

            render_info_y = 10
            for key in self.render_info:
                info_font = pygame.font.Font(None, 18)
                # 渲染文本到 Surface 对象
                text_surface = info_font.render(f'{key}: {self.render_info[key]}', True, (0, 0, 0))
                # 获取文本区域的矩形
                text_rect = text_surface.get_rect()
                text_rect.topleft = (10, render_info_y)
                self.screen.blit(text_surface, text_rect)  # 绘制文本到屏幕上
                render_info_y += 20

            self.battle_area.render(self.screen)
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()

    def should_render(self):
        if self.render_mode == 'human':
            time_passed = pygame.time.get_ticks() - self.last_render_time
            # print('should_render', time_passed, (1000 / self.metadata["render_fps"]))
            return time_passed > (1000 / self.metadata["render_fps"])
        else:
            return False

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
