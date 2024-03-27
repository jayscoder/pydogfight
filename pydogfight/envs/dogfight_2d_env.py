from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import pygame.time
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.error import DependencyNotInstalled
from pydogfight.core.world_obj import *
from pydogfight.core.options import Options
from pydogfight.core.battle_area import BattleArea
import time


class Dogfight2dEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # "render_fps"  : 50,
    }

    def __init__(self,
                 options: Options = Options(),
                 render_mode: str | None = None):
        super().__init__()
        assert len(options.red_agents) > 0 and len(options.blue_agents) > 0
        self.render_mode = render_mode
        self.options = options

        # (action_type, x, y)
        self.action_space_low = np.array([0, -options.game_size[1] / 2, -options.game_size[1] / 2])
        self.action_space_high = np.array([len(Actions), -options.game_size[1] / 2, -options.game_size[1] / 2])

        self.action_space = gym.spaces.Box(
                low=np.tile([0, -int(options.game_size[0] / 2), -int(options.game_size[1] / 2)],
                            (len(options.agents), 1)),
                high=np.tile([0, int(options.game_size[0] / 2), int(options.game_size[1] / 2)],
                             (len(options.agents), 1)),
                shape=(len(self.options.agents), 3),
                dtype=np.int32)

        # color, destroyed, x, y, psi, speed, missile_count, fuel, radar_radius
        self.observation_space = gym.spaces.Box(
                low=-options.game_size[0] / 2,
                high=options.game_size[1] / 2,
                shape=(len(options.red_agents) + len(options.blue_agents) + 10, 11),  # 最多同时记录所有飞机和10个导弹的信息
                dtype=np.float32)

        self.screen = None
        self.clock: pygame.time.Clock | None = None
        self.isopen = True

        self.battle_area = BattleArea(options=options, render_mode=self.render_mode)

        self.last_render_time = 0
        self.last_update_time = 0
        self.step_count = 0
        self.accum_reward = {
            'red' : 0,
            'blue': 0
        }  # 累积奖励
        self.render_info = { }

        # TODO 敌人的heatmap

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
        super().reset(seed=seed)
        self.step_count = 0
        self.accum_reward = {
            'red' : 0,
            'blue': 0
        }
        self.battle_area.reset()
        self.render_info['reward'] = 0

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                        self.options.screen_size,
                        self.options.pygame_mode)
                self.screen.fill((255, 255, 255))
                pygame.display.set_caption("dogfight")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self.render()
        return self.gen_obs(), { }

    def gen_obs(self):
        obs = np.zeros(self.observation_space.shape)
        i = 0
        for name in self.options.agents:
            obj = self.battle_area.get_obj(name)
            assert isinstance(obj, Aircraft)
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

        for obj in self.battle_area.objs.values():
            if not isinstance(obj, Missile):
                continue
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
        :param agent_name:
        :return:
        """
        agent_index = 0
        for i, name in enumerate(self.options.agents):
            if name == agent_name:
                agent_index = i
                break
        observation = self.gen_obs()
        # # 隐藏掉雷达探测范围以外的obs，并且移除一些无法获取的信息
        self_status = observation[agent_index]
        new_obs = np.zeros(observation.shape)
        new_obs[0] = observation[agent_index]  # 第一个是自己
        index = 1
        for i in range(observation.shape[0]):
            if i == agent_index:
                continue
            dis = np.linalg.norm(observation[i, 3:5] - self_status[3:5])
            if dis > self_status[9]:
                # 在自己的雷达探测范围之外
                continue
            if observation[i][2] == 1:
                # 被摧毁了
                continue
            if observation[i][0] == OBJECT_TO_IDX['aircraft']:
                # 飞机
                new_obs[index, :] = observation[i]
                if new_obs[index, 1] != self_status[1]:
                    # 是敌人
                    new_obs[index, 8] = -1  # fuel不知道
                    # new_obs[index, 9] = -1  # radar_radius不知道
                    new_obs[index, 10] = -1  # missile_count不知道
                index += 1
            elif observation[i][0] == OBJECT_TO_IDX['missile']:
                # 导弹
                new_obs[index, :] = observation[i]
                if new_obs[index, 1] != self_status[1]:
                    new_obs[index, 8] = -1
                index += 1
        return new_obs

    def gen_info(self) -> dict:
        info = {
            'truncated' : self.time > self.options.max_duration,
            'terminated': False,
            'winner'    : ''
        }
        remain_count = self.battle_area.remain_count

        if remain_count['missile']['red'] + remain_count['missile']['blue'] > 0:
            return info

        if remain_count['aircraft']['red'] == 0 or remain_count['aircraft']['blue'] == 0:
            if remain_count['aircraft']['red'] > 0:
                info['winner'] = 'red'
            elif remain_count['aircraft']['blue'] > 0:
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
        assert self.options.delta_time > 0
        assert self.options.update_interval >= self.options.delta_time
        for _ in range(int(self.options.update_interval / self.options.delta_time)):
            self.battle_area.update(delta_time=self.options.delta_time)

        if self.render_mode == 'human':
            self.last_update_time = time.time()
            self.render_info['time'] = self.time

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self.put_action(action)
        info = self.gen_info()
        # 注意这里返回的状态是更新动作之前的（想获取更新动作之后的状态需要手动调用update）
        return self.gen_obs(), self.gen_reward(color=self.options.self_side), info['terminated'], info[
            'truncated'], info

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
            event = pygame.event.poll()
            if event is not None:
                if event.type == pygame.VIDEORESIZE:
                    self.options.screen_size = event.size
                    # 调整尺寸

            self.screen.fill((255, 255, 255))

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
            render_rect(self.options, screen=self.screen, rect=self.options.safe_boundary, color='grey')

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

    def gen_reward(self, color: str = ''):
        if color == '':
            color = self.options.self_side
        info = self.gen_info()
        reward = (self.options.time_punish_reward * self.time) - self.accum_reward[color]
        self.accum_reward[color] += reward

        if info['winner'] != '':
            # 某一方获胜了
            if self.options.self_side == info['winner']:
                reward = self.options.win_reward
            elif info['winner'] == 'draw':
                reward = self.options.draw_reward
            else:
                reward = self.options.lose_reward

        if self.render_mode == 'human':
            for c in ['red', 'blue']:
                self.render_info[f'reward_{c}'] = self.accum_reward[c]

        return reward
