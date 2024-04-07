from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from pydogfight.core.world_obj import *
from pydogfight.core.options import Options
from pydogfight.core.battle_area import BattleArea
from pydogfight.core.actions import Actions
from pydogfight.utils.common import cal_relative_polar_wpt
import time
import threading

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
                            (len(options.agents), 1)),
                high=np.tile([0, int(options.game_size[0] / 2), int(options.game_size[1] / 2)],
                             (len(options.agents), 1)),
                shape=(len(self.options.agents), 3),
                dtype=np.int32)

        # color, destroyed, x, y, psi, speed, missile_count, fuel, radar_radius
        self.observation_space = gym.spaces.Box(
                low=-options.game_size[0] / 2,
                high=options.game_size[0] / 2,
                shape=(len(options.red_agents) + len(options.blue_agents) + 13, 11),  # 最多同时记录所有飞机、基地、牛眼和10个导弹的信息
                dtype=np.float32)

        self.screen = None
        # self.clock # pygame.time.Clock
        self.isopen = True
        self.paused = False  # 是否暂停

        self.battle_area = BattleArea(options=options, render_mode=self.render_mode)

        self.last_render_time = 0
        self.last_update_time = 0
        self.step_count = 0
        self.accum_reward = {
            'red' : 0,
            'blue': 0
        }  # 累积奖励
        self.render_info = { }  # 渲染在屏幕上的信息

        self.game_info = {
            'red_wins'   : 0,
            'blue_wins'  : 0,
            'red_reward' : 0,
            'blue_reward': 0,
            'draw'       : 0,
            'round'      : 0,  # 第几轮
            'truncated'  : 0,
            'terminated' : 0,
        }  # 游戏对战累积数据，在reset的时候更新
        # TODO 敌人的heatmap

    @property
    def time(self):
        return self.battle_area.time

    def get_agent(self, name) -> Aircraft:
        obj = self.battle_area.get_obj(name)
        assert isinstance(obj, Aircraft)
        return obj

    def get_home(self, color) -> Home:
        if color == 'red':
            obj = self.battle_area.get_obj(self.options.red_home)
        else:
            obj = self.battle_area.get_obj(self.options.blue_home)
        assert isinstance(obj, Home)
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
            self.game_info['draw'] += 1

        if info['truncated']:
            self.game_info['truncated'] += 1
        if info['terminated']:
            self.game_info['terminated'] += 1

        self.game_info['red_reward'] += self.accum_reward['red']
        self.game_info['blue_reward'] += self.accum_reward['blue']
        self.game_info['round'] += 1

        super().reset(seed=seed)

        self.step_count = 0
        self.accum_reward = {
            'red' : 0,
            'blue': 0
        }

        self.battle_area.reset()
        self.render_info['reward'] = 0

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
        agent = self.get_agent(name=agent_name)
        obs = np.zeros(self.observation_space.shape)

        obs[0] = [
            OBJECT_TO_IDX[agent.type],  # 0
            COLOR_TO_IDX[agent.color],  # 1
            int(agent.destroyed),  # 2
            0,  # r 3
            0,  # theta 4
            0,  # psi 5
            agent.speed,  # 6
            agent.turn_radius,  # 7
            agent.fuel,  # 8
            agent.radar_radius,  # 9
            agent.missile_count,  # 10
        ]
        i = 1
        # 隐藏掉雷达探测范围以外的obs，并且移除一些无法获取的信息
        # 飞机
        for obj in self.battle_area.agents:
            if obj.name == agent_name:
                continue
            if not self.options.obs_ignore_radar and agent.distance(obj) > agent.radar_radius:
                # 在自己的雷达范围之外
                continue
            if self.options.obs_ignore_destroyed and obj.destroyed:
                continue
            rel_polar_wpt = cal_relative_polar_wpt(wpt1=(agent.x, agent.y, agent.psi), wpt2=(obj.x, obj.y, obj.psi))
            obs[i, :] = [
                OBJECT_TO_IDX[obj.type],  # 0
                COLOR_TO_IDX[obj.color],  # 1
                int(obj.destroyed),  # 2
                rel_polar_wpt[0],  # r 3
                rel_polar_wpt[1],  # theta 4
                rel_polar_wpt[2],  # psi 5
                obj.speed,  # 6
                obj.turn_radius,  # 7
                obj.fuel,  # 8
                obj.radar_radius,  # 9
                obj.missile_count,  # 10
            ]

            if self.options.obs_ignore_enemy_detail and obj.color != agent.color:
                # 不知道敌机的油量和导弹数
                obs[i, 8] = -1
                obs[i, 10] = -1

            i += 1

        # 基地
        for obj in self.battle_area.homes:
            rel_polar_wpt = cal_relative_polar_wpt(wpt1=(agent.x, agent.y, agent.psi), wpt2=(obj.x, obj.y, obj.psi))
            obs[i, :] = [
                OBJECT_TO_IDX[obj.type],  # 0
                COLOR_TO_IDX[obj.color],  # 1
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
            COLOR_TO_IDX[bullseye.color],  # 1
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
                COLOR_TO_IDX[obj.color],  # 1
                int(obj.destroyed),  # 2
                rel_polar_wpt[0],  # 3
                rel_polar_wpt[1],  # 4
                rel_polar_wpt[2],  # 5
                obj.speed,  # 6
                obj.turn_radius,  # 7
                obj.fuel,  # 8
                0,  # 9
                0,  # 10
            ]
            if self.options.obs_ignore_missile_fuel:
                obs[i, 8] = -1
            if self.options.obs_ignore_enemy_detail and obj.color != agent.color:
                # 不知道敌方导弹的剩余油量
                obs[i, 8] = -1

            i += 1

        return obs

    def gen_info(self) -> dict:
        info = {
            'truncated' : False,
            'terminated': False,
            'winner'    : ''
        }
        remain_count = self.battle_area.remain_count

        if remain_count['missile']['red'] + remain_count['missile']['blue'] > 0:
            return info

        info['truncated'] = self.time >= self.options.max_duration

        if remain_count['aircraft']['red'] == 0 or remain_count['aircraft']['blue'] == 0:
            if remain_count['aircraft']['red'] > 0:
                info['winner'] = 'red'
            elif remain_count['aircraft']['blue'] > 0:
                info['winner'] = 'blue'
            else:
                info['winner'] = 'draw'
            info['terminated'] = True
        elif info['truncated']:
            info['winner'] = 'draw'  # 不这么做的话，强化学习就会一直停留原地打转，等待敌机找上门或失误

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
            self.battle_area.update()

        if self.render_mode == 'human':
            self.last_update_time = time.time()
            self.render_info['time'] = self.time

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        obs, reward, terminated, truncated, info = env.step(action)
        :param action:
        :return:
        """
        self.step_count += 1
        self.put_action(action)
        game_time = self.time

        self.update() # 更新环境

        time_passed = self.time - self.last_update_time



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

    def gen_reward(self, color: str = ''):
        if color == '':
            color = self.options.self_side
        info = self.gen_info()
        reward = (self.options.time_punish_reward * self.time) - self.accum_reward[color]

        if info['winner'] != '':
            # 某一方获胜了
            if color == info['winner']:
                reward = self.options.win_reward
            elif info['winner'] == 'draw':
                reward = self.options.draw_reward
            else:
                reward = self.options.lose_reward

        if self.render_mode == 'human':
            for c in ['red', 'blue']:
                self.render_info[f'reward_{c}'] = self.accum_reward[c]

        self.accum_reward[color] += reward
        return reward
