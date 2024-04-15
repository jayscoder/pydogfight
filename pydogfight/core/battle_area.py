from __future__ import annotations

from pydogfight.core.world_obj import *
from pydogfight.core.options import Options
from collections import defaultdict
import typing


class BattleArea:
    def __init__(self, options: Options, render_mode: str = 'rgb_array'):
        self.options = options
        self.size = options.game_size
        self.time = 0  # 对战时长
        self.accum_time = 0  # 对战累积时长
        self.objs: dict[str, WorldObj] = { }
        self.render_mode = render_mode
        self.cache = { }  # 缓存
        self.stats = {
            'episode': 0,
            'red'    : {
                'win'      : 0,
                'lose'     : 0,
                'draw'     : 0,
                'win_rate' : 0.0,
                'lose_rate': 0.0,
                'draw_rate': 0.0
            },
            'blue'   : {
                'win'      : 0,
                'lose'     : 0,
                'draw'     : 0,
                'win_rate' : 0.0,
                'lose_rate': 0.0,
                'draw_rate': 0.0
            },
            'agent'  : { }
        }

    @property
    def episode(self):
        return self.stats['episode']

    def episode_end(self):
        self.accum_time += self.time
        self.stats['episode'] += 1
        winner = self.winner
        if winner == '':
            winner = 'draw'
        self.stats['winner'] = winner

        if winner == 'red':
            self.stats['red']['win'] += 1
            self.stats['blue']['lose'] += 1
        elif winner == 'blue':
            self.stats['blue']['win'] += 1
            self.stats['red']['lose'] += 1
        else:
            self.stats['red']['draw'] += 1
            self.stats['blue']['draw'] += 1

        for color in ['red', 'blue']:
            for k in ['win', 'lose', 'draw']:
                self.stats[color][f'{k}_rate'] = self.stats[color][k] / self.episode

        new_stats = {
            'red'  : { },
            'blue' : { },
            'agent': { }
        }

        KEYS = [
            'destroyed_count',
            'missile_fired_count',
            'missile_fire_fail_count',
            'missile_hit_self_count',
            'missile_hit_enemy_count',
            'missile_miss_count',
            'missile_evade_success_count',
            'home_returned_count',
            'missile_count',
            'missile_depletion_count',
            'aircraft_collided_count',
        ]

        for agent in self.agents:
            if agent.name not in new_stats['agent']:
                new_stats['agent'][agent.name] = { }

            for key in KEYS:
                new_stats['agent'][agent.name][key] = getattr(agent, key)
                dict_incr(new_stats[agent.color], key=key, value=getattr(agent, key))
        merge_tow_dicts(new_stats, self.stats)

    def episode_start(self):
        self.time = 0
        self.objs.clear()
        self.cache.clear()

        # 加载

        assert self.options.red_home != ''
        assert self.options.blue_home != ''
        assert len(self.options.red_agents) > 0
        assert len(self.options.blue_agents) > 0
        red_home_pos = self.options.generate_home_init_position(color='red')
        blue_home_pos = self.options.generate_home_init_position(color='blue')

        self.add_obj(
                Home(
                        name=self.options.red_home,
                        options=self.options,
                        color='red',
                        waypoint=Waypoint.build(x=red_home_pos[0], y=red_home_pos[1], psi=0)
                )
        )

        self.add_obj(
                Home(
                        name=self.options.blue_home,
                        options=self.options,
                        color='blue',
                        waypoint=Waypoint.build(x=blue_home_pos[0], y=blue_home_pos[1], psi=0)
                )
        )

        self.add_obj(
                Bullseye(options=self.options)
        )

        for name in self.options.blue_agents:
            # 随机生成飞机位置
            wpt = self.options.generate_aircraft_init_waypoint(color='blue', home_position=blue_home_pos)
            self.add_obj(Aircraft(
                    name=name,
                    options=self.options,
                    color='blue',
                    waypoint=Waypoint(data=wpt)))

        for name in self.options.red_agents:
            # 随机生成飞机位置
            wpt = self.options.generate_aircraft_init_waypoint(color='red', home_position=red_home_pos)
            self.add_obj(Aircraft(
                    name=name,
                    options=self.options,
                    color='red',
                    waypoint=Waypoint(data=wpt)))

    def add_obj(self, obj: WorldObj):
        self.objs[obj.name] = obj
        obj.attach(battle_area=self)

    def get_obj(self, name: str) -> WorldObj:
        return self.objs[name]

    def remove_obj(self, obj: WorldObj) -> None:
        del self.objs[obj.name]

    @property
    def agents(self) -> list[Aircraft]:
        return list(filter(lambda agent: isinstance(agent, Aircraft), self.objs.values()))

    @property
    def missiles(self) -> list[Missile]:
        return list(filter(lambda obj: isinstance(obj, Missile), self.objs.values()))

    @property
    def homes(self) -> list[Home]:
        return list(filter(lambda obj: isinstance(obj, Home), self.objs.values()))

    @property
    def bullseye(self) -> Bullseye:
        obj = self.objs.get('bullseye')
        assert isinstance(obj, Bullseye)
        return obj

    def get_home(self, color: str) -> Home:
        if color == 'red':
            obj = self.get_obj(self.options.red_home)
        else:
            obj = self.get_obj(self.options.blue_home)
        assert isinstance(obj, Home)
        return obj

    def get_agent(self, agent_name: str) -> Aircraft:
        obj = self.get_obj(agent_name)
        assert isinstance(obj, Aircraft)
        return obj

    def render(self, screen):
        for obj in self.objs.values():
            obj.render(screen)

    ### 战场环境更新，每一轮每个物体只消费一个行为 ###
    def update(self):
        """
        :return:
        """
        not_destroyed_objs = [obj for obj in self.objs.values() if not obj.destroyed]
        for obj in not_destroyed_objs:
            obj.update(delta_time=self.options.delta_time)

        # 检查碰撞，通过缓存来确保只触发一次（需要先进入非碰撞状态才能触发碰撞）
        for i in range(len(not_destroyed_objs)):
            obj_1 = not_destroyed_objs[i]
            if obj_1.collision_radius <= 0:
                continue
            for j in range(i + 1, len(not_destroyed_objs)):
                obj_2 = not_destroyed_objs[j]

                if obj_2.collision_radius <= 0:
                    continue

                new_collided = obj_1.will_collide(obj_2)
                collided_key = f'collided-{obj_1.name}-{obj_2.name}'
                old_collided = self.cache.get(collided_key, None)
                if new_collided and old_collided == False:
                    obj_1.on_collision(obj_2)
                    obj_2.on_collision(obj_1)
                self.cache[collided_key] = new_collided

        # 移除掉被摧毁的导弹
        destroyed_objs = [obj for obj in self.objs.values() if obj.destroyed]

        for obj in destroyed_objs:
            if isinstance(obj, Missile):
                self.remove_obj(obj)

        self.time += self.options.delta_time

    @property
    def remain_count(self) -> dict:
        """
        获取剩余实体数量
        :return:
        """
        count = {
            'aircraft': {
                'red' : 0,
                'blue': 0
            },
            'missile' : {
                'red' : 0,
                'blue': 0
            },
            'home'    : {
                'red' : 0,
                'blue': 0
            }
        }
        for obj in self.objs.values():
            if obj.destroyed:
                continue
            if obj.type not in count:
                count[obj.type] = {
                    'red' : 0,
                    'blue': 0
                }
            if obj.color not in count[obj.type]:
                count[obj.type][obj.color] = 0
            count[obj.type][obj.color] += 1
        return count

    @property
    def winner(self) -> str:
        """
        计算胜利方
        Returns:
            red: 红方获胜
            blue: 蓝方获胜
            draw: 平局

        """
        remain_count = self.remain_count

        if self.options.indestructible:
            # 无敌模式下，用到达结束时间时双方的被摧毁数量来判断胜负
            if self.time < self.options.max_duration:
                return ''

            destroyed_count = {
                'red' : 0,
                'blue': 0
            }

            for agent in self.agents:
                destroyed_count[agent.color] += agent.destroyed_count

            if destroyed_count['red'] > destroyed_count['blue']:
                # 红方被摧毁次数多，蓝方获胜
                return 'blue'
            elif destroyed_count['blue'] > destroyed_count['red']:
                return 'red'
            else:
                return 'draw'
        else:
            if self.time >= self.options.max_duration:
                # 超时就认为是平局
                return 'draw'

            if remain_count['aircraft']['red'] == 0 and remain_count['aircraft']['blue'] == 0:
                return 'draw'

            if remain_count['missile']['red'] + remain_count['missile']['blue'] > 0:
                return ''

            if remain_count['aircraft']['red'] == 0:
                if remain_count['missile']['red'] > 0:
                    return ''
                return 'blue'

            if remain_count['aircraft']['blue'] == 0:
                if remain_count['missile']['blue'] > 0:
                    return ''
                return 'red'
        return ''

    def detect_missiles(self, agent_name: str, ignore_radar: bool = False, only_enemy: bool = True) -> list[Missile]:
        """
        检测来袭导弹
        Args:
            agent_name: 我方战机名称
            ignore_radar: 忽略雷达
            only_enemy: 只检测敌机的导弹

        Returns: 来袭导弹，按照距离从小到大排序

        """
        missiles = []
        agent = self.get_obj(agent_name)
        assert isinstance(agent, Aircraft)
        for obj in self.objs.values():
            if obj.name == agent_name or not isinstance(obj, Missile) or obj.destroyed:
                continue
            if not ignore_radar and not agent.in_radar_range(obj):
                # 不在雷达范围内
                continue
            if only_enemy and obj.color == agent.color:
                # 不检测相同战队的导弹
                continue
            missiles.append((obj, obj.distance(agent)))
        return list(map(lambda it: it[0], sorted(missiles, key=lambda it: it[1])))

    def detect_aircraft(self, agent_name: str, ignore_radar: bool = False, only_enemy: bool = True) -> list[Aircraft]:
        """
        检测来袭飞机
        Args:
            agent_name: 我方战机名称
            ignore_radar: 忽略雷达
            only_enemy: 只检测敌机

        Returns: 来袭飞机，按照距离从小到大排序

        """
        aircraft_items = []
        agent = self.get_obj(agent_name)
        assert isinstance(agent, Aircraft)
        for obj in self.objs.values():
            if obj.name == agent_name or not isinstance(obj, Aircraft) or obj.destroyed:
                continue
            if not ignore_radar and not agent.in_radar_range(obj):
                # 不在雷达范围内
                continue
            if only_enemy and obj.color == agent.color:
                # 只检测敌机
                continue
            aircraft_items.append((obj, obj.distance(agent)))
        return list(map(lambda it: it[0], sorted(aircraft_items, key=lambda it: it[1])))

    def find_nearest_enemy(self, agent_name: str, ignore_radar: bool = False) -> Aircraft | None:
        """
        找到距离最近的敌人
        :param agent_name:
        :param ignore_radar: 是否忽略雷达因素（设为false则只会返回雷达范围内的敌机）
        :return:
        """
        agent = self.objs[agent_name]
        assert isinstance(agent, Aircraft)
        min_dis = float('inf')
        enemy = None
        for obj in self.objs.values():
            if obj.destroyed or not isinstance(obj, Aircraft):
                continue
            if obj.color == agent.color:
                continue

            dis = obj.distance(agent)
            if not ignore_radar and dis > agent.radar_radius:
                continue

            if dis < min_dis:
                min_dis = dis
                enemy = obj

        return enemy
