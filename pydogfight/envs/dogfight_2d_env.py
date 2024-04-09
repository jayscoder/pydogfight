from __future__ import annotations

import typing
from typing import Any, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from pydogfight.core.world_obj import *
from pydogfight.core.options import Options
from pydogfight.core.battle_area import BattleArea
from pydogfight.core.actions import Actions
from pydogfight.utils.common import cal_relative_polar_wpt
import time
import threading
import asyncio


class Dogfight2dEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # "render_fps"  : 50,
    }

    def __init__(self,
                 options: Options = Options(),
                 render_mode: str | None = None):
        super().__init__()
        options.validate()
        self.render_mode = render_mode
        self.options = options

        # (action_type, x, y)
        self.action_space_low = np.array([0, -options.game_size[1] / 2, -options.game_size[1] / 2])
        self.action_space_high = np.array([len(Actions), -options.game_size[1] / 2, -options.game_size[1] / 2])

        self.action_space = gym.spaces.Box(
                low=np.tile([0, -int(options.game_size[0] / 2), -int(options.game_size[1] / 2)],
                            (len(options.agents()), 1)),
                high=np.tile([len(Actions), int(options.game_size[0] / 2), int(options.game_size[1] / 2)],
                             (len(options.agents()), 1)),
                shape=(len(self.options.agents()), 3),
                dtype=np.int32)

        self.agent_action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1]),
                shape=(3,),
        )

        # type, color, destroyed, x, y, psi, speed, missile_count, fuel, radar_radius
        self.observation_space = gym.spaces.Box(
                low=-options.game_size[0] / 2,
                high=options.game_size[0] / 2,
                shape=(len(options.agents()) + 3 + 5, 11),  # 最多同时记录所有飞机、基地、牛眼和5个导弹的信息
                dtype=np.float32)

        # type, is_enemy, destroyed, r, theta, psi, speed, missile_count, fuel, radar_radius
        self.agent_observation_space = gym.spaces.Box(
                low=-10,
                high=10,
                shape=(len(options.agents()) + 3 + 5, 11),  # 最多同时记录所有飞机、基地、牛眼和5个导弹的信息
                dtype=np.float32)

        self.screen = None
        # self.clock # pygame.time.Clock
        self.isopen = True
        self.paused = False  # 是否暂停

        self.battle_area = BattleArea(options=options, render_mode=self.render_mode)

        self.last_render_time = 0
        self.last_update_time = 0
        self.step_count = 0

        self.render_info = { }  # 渲染在屏幕上的信息

        self.game_info = {
            'red_wins'         : 0,
            'blue_wins'        : 0,
            'draws'            : 0,  # 平局几次
            'round'            : 0,  # 第几轮
            'red_accum_reward' : 0,  # 红方累积奖励
            'blue_accum_reward': 0,  # 蓝方累积奖励
            'truncated_count'  : 0,
            'terminated_count' : 0,
        }  # 游戏对战累积数据，在reset的时候更新
        # TODO 敌人的heatmap

        self.update_event = { }
        for agent_name in self.options.agents():
            self.update_event[agent_name] = threading.Event()

        self.agent_memory = { }

        self.after_update_handlers: list[typing.Callable[['Dogfight2dEnv'], None]] = []
        self.after_reset_handlers: list[typing.Callable[['Dogfight2dEnv'], None]] = []

    def add_after_update_handler(self, handler: typing.Callable[['Dogfight2dEnv'], None]):
        self.after_update_handlers.append(handler)

    def add_after_reset_handler(self, handler: typing.Callable[['Dogfight2dEnv'], None]):
        self.after_reset_handlers.append(handler)

    @property
    def time(self):
        return self.battle_area.time

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
        info = self.gen_info()
        if info['winner'] == 'red':
            self.game_info['red_wins'] += 1
        elif info['winner'] == 'blue':
            self.game_info['blue_wins'] += 1
        elif info['winner'] == 'draw':
            self.game_info['draws'] += 1

        if info['truncated']:
            self.game_info['truncated_count'] += 1
        if info['terminated']:
            self.game_info['terminated_count'] += 1

        self.game_info['round'] += 1
        self.game_info['red_accum_reward'] += info['red_reward']
        self.game_info['blue_accum_reward'] += info['blue_reward']

        super().reset(seed=seed)

        self.step_count = 0

        self.battle_area.reset()

        for agent_name in self.update_event:
            self.update_event[agent_name].clear()  # 重置更新事件

        if self.render_mode == "human":
            import pygame
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                        self.options.screen_size,
                        pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
                self.screen.fill((255, 255, 255))
                pygame.display.set_caption("dogfight")
            # if self.clock is None:
            #     self.clock = pygame.time.Clock()
            self.render()

        for handler in self.after_reset_handlers:
            handler(self)

        return self.gen_obs(), self.gen_info()

    def gen_obs(self):
        obs = np.zeros(self.observation_space.shape)
        i = 0
        # 飞机
        for obj in self.battle_area.agents:
            obs[i, :] = [
                OBJECT_TO_IDX[obj.type],  # 0
                COLOR_TO_IDX[obj.color],  # 1
                int(obj.destroyed),  # 2
                obj.x,  # 3
                obj.y,  # 4
                obj.psi,  # 5
                obj.speed,  # 6
                obj.turn_radius,  # 7
                obj.fuel,  # 8
                obj.radar_radius,  # 9
                obj.missile_count,  # 10
            ]
            i += 1

        # 基地
        for obj in self.battle_area.homes:
            obs[i, :] = [
                OBJECT_TO_IDX[obj.type],  # 0
                COLOR_TO_IDX[obj.color],  # 1
                int(obj.destroyed),  # 2
                obj.x,  # 3
                obj.y,  # 4
                0,  # 5
                0,  # 6
                0,  # 7
                0,  # 8
                0,  # 9
                0,  # 10
            ]
            i += 1

        # 导弹
        for obj in self.battle_area.missiles:
            if obj.destroyed:
                continue
            if i >= len(obs):
                break
            obs[i, :] = [
                OBJECT_TO_IDX[obj.type],  # 0
                COLOR_TO_IDX[obj.color],  # 1
                int(obj.destroyed),  # 2
                obj.x,  # 3
                obj.y,  # 4
                obj.psi,  # 5
                obj.speed,  # 6
                obj.turn_radius,  # 7
                obj.fuel,  # 8
                0,  # 9
                0,  # 10
            ]
            i += 1

        return obs

    def gen_agent_obs(self, agent_name: str):
        """
        获取agent视角的obs
        注意这里的坐标用相对极坐标来表示
        :param agent_name:
        :return:
        """
        if agent_name not in self.agent_memory:
            self.agent_memory[agent_name] = { }

        agent = self.get_agent(name=agent_name)
        obs = np.zeros(self.observation_space.shape)

        obs[0] = [
            OBJECT_TO_IDX[agent.type],  # 0
            0,  # is_enemy 1
            int(agent.destroyed),  # 2
            0,  # r 3
            0,  # theta 4
            0,  # psi 5
            agent.speed / agent.radar_radius,  # 6
            agent.turn_radius / agent.radar_radius,  # 7
            agent.fuel / self.options.aircraft_fuel_capacity,  # 8
            agent.radar_radius / agent.radar_radius,  # 9
            agent.missile_count / self.options.aircraft_missile_count,  # 10
        ]
        i = 1
        # 隐藏掉雷达探测范围以外的obs，并且移除一些无法获取的信息
        # 飞机
        for obj in self.battle_area.agents:
            if obj.name == agent_name:
                continue
            if self.options.obs_ignore_destroyed and obj.destroyed:
                continue

            def _obj_obs(obj: Aircraft, is_memory: bool):
                rel_polar_wpt = cal_relative_polar_wpt(wpt1=(agent.x, agent.y, agent.psi), wpt2=(obj.x, obj.y, obj.psi))
                obj_obs = [
                    OBJECT_TO_IDX[obj.type],  # 0
                    int(obj.color != agent.color),  # 1
                    -1 if is_memory else 0,  # is_destroyed 2
                    rel_polar_wpt[0],  # r 3
                    rel_polar_wpt[1],  # theta 4
                    rel_polar_wpt[2],  # psi 5
                    obj.speed / agent.radar_radius,  # 6
                    obj.turn_radius / agent.radar_radius,  # 7
                    obj.fuel / self.options.aircraft_fuel_capacity,  # 8
                    obj.radar_radius / agent.radar_radius,  # 9
                    obj.missile_count / self.options.aircraft_missile_count,  # 10
                ]
                if obj.color != agent.color:
                    if self.options.obs_ignore_enemy_fuel:
                        # 不知道敌机的油量和导弹数
                        obj_obs[8] = -1
                    if self.options.obs_ignore_enemy_missile_count:
                        # 不知道敌机的剩余导弹数
                        obj_obs[10] = -1
                return obj_obs

            if not self.options.obs_ignore_radar and agent.distance(obj) > agent.radar_radius:
                # 在自己的雷达范围之外

                if self.options.obs_allow_memory:
                    # 允许使用记忆
                    if obj.name in self.agent_memory[agent_name]:
                        obj = self.agent_memory[agent_name][obj.name]
                        obs[i, :] = _obj_obs(obj, is_memory=True)

                continue
            obs[i, :] = _obj_obs(obj, is_memory=False)
            self.agent_memory[agent_name][obj.name] = agent.__copy__()
            i += 1

        # 基地默认是知道的（不考虑雷达）
        for obj in self.battle_area.homes:
            rel_polar_wpt = cal_relative_polar_wpt(wpt1=(agent.x, agent.y, agent.psi), wpt2=(obj.x, obj.y, obj.psi))
            obs[i, :] = [
                OBJECT_TO_IDX[obj.type],  # 0
                int(obj.color != agent.color),  # 1
                int(obj.destroyed),  # 2
                rel_polar_wpt[0],  # 3
                rel_polar_wpt[1],  # 4
                0,  # 5
                0,  # 6
                0,  # 7
                0,  # 8
                0,  # 9
                0,  # 10
            ]
            i += 1

        # 牛眼
        bullseye = self.battle_area.bullseye
        rel_polar_wpt = cal_relative_polar_wpt(wpt1=(agent.x, agent.y, agent.psi),
                                               wpt2=(bullseye.x, bullseye.y, bullseye.psi))
        obs[i, :] = [
            OBJECT_TO_IDX[bullseye.type],  # 0
            0,  # 1
            int(bullseye.destroyed),  # 2
            rel_polar_wpt[0],  # 3
            rel_polar_wpt[1],  # 4
            0,  # 5
            0,  # 6
            0,  # 7
            0,  # 8
            0,  # 9
            0,  # 10
        ]
        i += 1

        # 导弹
        for obj in self.battle_area.missiles:
            if obj.destroyed:
                continue
            if i >= len(obs):
                break
            rel_polar_wpt = cal_relative_polar_wpt(wpt1=(agent.x, agent.y, agent.psi), wpt2=(obj.x, obj.y, obj.psi))
            obs[i, :] = [
                OBJECT_TO_IDX[obj.type],  # 0
                int(obj.color != agent.color),  # 1
                int(obj.destroyed),  # 2
                rel_polar_wpt[0],  # 3
                rel_polar_wpt[1],  # 4
                rel_polar_wpt[2],  # 5
                obj.speed / agent.radar_radius,  # 6
                obj.turn_radius / agent.radar_radius,  # 7
                obj.fuel / self.options.missile_fuel_capacity,  # 8
                0,  # 9
                0,  # 10
            ]

            if obj.color != agent.color:
                if self.options.obs_ignore_enemy_missile_fuel:
                    # 不知道敌方导弹的剩余油量
                    obs[i, 8] = -1

            i += 1

        return obs

    def gen_info(self) -> dict:
        info = {
            'truncated'   : self.time >= self.options.max_duration,
            'terminated'  : self.battle_area.winner != '',
            'winner'      : self.battle_area.winner,
            'time'        : self.time,
            'remain_count': self.battle_area.remain_count,
            'red_reward'  : self.gen_reward(color='red', previous=0),
            'blue_reward' : self.gen_reward(color='blue', previous=0)
        }
        return info

    def empty_action(self):
        return np.zeros(self.action_space.shape)

    def put_action(self, action):
        for i, act in enumerate(action):
            if act[0] == 0:
                continue
            agent_name = self.options.agents()[i]
            agent = self.battle_area.get_obj(agent_name)
            agent.put_action(act)

    def update(self):
        assert self.options.delta_time > 0
        assert self.options.update_interval >= self.options.delta_time
        for _ in range(int(self.options.update_interval / self.options.delta_time)):
            self.battle_area.update()

        if self.render_mode == 'human':
            self.last_update_time = time.time()
            self.render_info.update(self.gen_info())

        for agent_name in self.update_event:
            self.update_event[agent_name].set()  # 设置事件，表示更新完成

        for handler in self.after_update_handlers:
            handler(self)

    async def agent_step(self, agent_name: str, action: list | np.ndarray | tuple[float, float, float]):
        old_info = self.gen_info()
        agent = self.get_agent(agent_name)
        agent.put_action(action)
        self.update_event[agent_name].clear()  # 重置事件
        # 等待环境更新完成
        while not self.update_event[agent_name].is_set():
            pass
        info = self.gen_info()
        reward = self.gen_reward(color=agent.color, previous=old_info)
        return self.gen_obs(), reward, info['terminated'], info['truncated'], info

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        obs, reward, terminated, truncated, info = env.step(action)
        :param action: 所有动作
        :return:
        """
        self.step_count += 1
        self.put_action(action)
        old_info = self.gen_info()
        self.update()  # 更新环境
        info = self.gen_info()
        reward = self.gen_reward(color=self.options.self_side, previous=old_info)
        # 注意这里返回的状态是更新动作之前的（想获取更新动作之后的状态需要手动调用update）
        return self.gen_obs(), reward, info['terminated'], info['truncated'], info

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                    "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        # print(f'Render {self.duration} {self.last_render_time}')
        self.last_render_time = time.time()

        if self.render_mode == "human":
            self.screen.fill((255, 255, 255))

            play_pause_img = pygame_load_img('play.svg' if self.paused else 'pause.svg').convert_alpha()
            play_pause_img_rect = play_pause_img.get_rect()
            play_pause_img_rect.right = self.options.screen_size[0] - 10
            play_pause_img_rect.top = 10

            event = pygame.event.poll()
            if event is not None:
                if event.type == pygame.VIDEORESIZE:
                    self.options.screen_size = event.size
                    # 调整尺寸
                # 检测鼠标点击
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if play_pause_img_rect.collidepoint(event.pos):
                        self.paused = not self.paused  # 切换暂停状态

            render_info_y = 10
            for key in self.render_info:
                # 渲染文本到 Surface 对象
                render_text(
                        screen=self.screen,
                        text=f'{key}: {self.render_info[key]}',
                        topleft=(10, render_info_y),
                        text_size=18,
                )
                render_info_y += 20

            # 渲染安全区域
            # render_rect(self.options, screen=self.screen, rect=self.options.safe_boundary, color='grey')
            self.screen.blit(play_pause_img, play_pause_img_rect)
            self.battle_area.render(self.screen)
            pygame.event.pump()
            # self.clock.tick(self.options.render_fps)
            pygame.display.update()

    def should_render(self):
        if self.render_mode == 'human':
            time_passed = time.time() - self.last_render_time
            # print('should_render', time_passed, 'seconds')
            return time_passed >= (1 / self.options.render_fps)
        else:
            return False

    def should_update(self):
        if self.paused:
            return
        if self.render_mode == 'human':
            time_passed = (time.time() - self.last_update_time) * self.options.simulation_rate
            # print('should_update', time_passed, 'seconds')
            return time_passed >= self.options.update_interval
        else:
            return True

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def gen_reward(self, color: str, previous: dict | float | None):
        """
        生成某方的累积奖励
        Args:
            color:
            previous: 之前的奖励/info，填0代表返回累积奖励
        Returns:
        """
        reward = self.options.time_punish_reward * self.time  # 时间惩罚
        winner = self.battle_area.winner
        if winner != '':
            # 某一方获胜了
            if winner == color:
                reward += self.options.win_reward
            elif winner == 'draw':
                reward += self.options.draw_reward
            else:
                reward += self.options.lose_reward
        if previous is None:
            previous = 0
        if isinstance(previous, dict):
            previous = previous[f'{color}_reward']
        return reward - previous
