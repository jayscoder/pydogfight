from __future__ import annotations

import os.path
from typing import Any, Dict

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from pydogfight import *
from pydogfight.policy import Policy
from pydogfight.wrappers import AgentWrapper
import json
import pybts
import pydogfight

options = Options()
options.delta_time = 1  # 每次更新的间隔
options.self_side = 'red'
options.simulation_rate = 1000
options.policy_interval = 1
BT_BOARD_TRACK = True
MODEL_NAME = 'ppo_move'
MODEL_PATH = os.path.join('models', MODEL_NAME)
TEST_N = 100
BT_BOARD_TRACK_PREFIX = 'ppo_move'


class ModelGoToLocationTrainWrapper(AgentWrapper):
    """
    指定单个Agent飞行控制的强化学习训练的视角
    """

    def __init__(self, policies: list[Policy], env: Dogfight2dEnv, agent_name: str = ''):
        super().__init__(env=env, agent_name=agent_name)
        self.observation_space = env.observation_space
        # 定义连续动作空间，比如一个二维连续动作，每个维度的范围是-1到1，以自身为原点，雷达范围内的坐标点（放缩到[-1, 1]之间）
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.policies = policies

    def step(self, action):
        assert isinstance(self.env, Dogfight2dEnv)
        for p in self.policies:
            p.select_action()
            p.put_action()
        agent = self.env.get_agent(self.agent_name)

        relative_waypoint = agent.waypoint.relative_move(dx=action[0] * agent.radar_radius,
                                                         dy=action[1] * agent.radar_radius)

        agent.put_action((Actions.go_to_location, relative_waypoint.x, relative_waypoint.y))
        self.env.update()
        obs = self.env.gen_agent_obs(agent_name=self.agent_name)
        info = self.env.gen_info()
        reward = self.env.gen_reward(color=agent.color)
        return obs, reward, info['terminated'], info['truncated'], info


def create_bt_greedy_policy(env: Dogfight2dEnv, agent_name: str, filepath: str):
    filename = os.path.basename(filepath).replace('.xml', '')
    tree = pybts.Tree(
            root=pydogfight.policy.BTPolicyBuilder().build_from_file(filepath),
            name=os.path.join(BT_BOARD_TRACK_PREFIX, agent_name, filename))
    policy = pydogfight.policy.BTPolicy(
            env=env,
            tree=tree,
            agent_name=agent_name,
            update_interval=options.policy_interval
    )

    if BT_BOARD_TRACK:
        board = pybts.Board(tree=policy.tree, log_dir='logs')
        tree.add_post_tick_handler(lambda t: board.track({
            'env_time': env.time,
            **env.render_info,
            agent_name: env.get_agent(agent_name).to_dict(),
        }))
        board.clear()
    return policy


def create_bt_model_policy(env: Dogfight2dEnv, agent_name: str):
    tree = pybts.Tree(
            root=pydogfight.policy.BTPPOGoToLocationModel(
                    model=MODEL_PATH
            ),
            name=os.path.join(BT_BOARD_TRACK_PREFIX, MODEL_NAME))

    policy = pydogfight.policy.BTPolicy(
            env=env,
            tree=tree,
            agent_name=agent_name,
            update_interval=options.policy_interval
    )

    if BT_BOARD_TRACK:
        board = pybts.Board(tree=policy.tree, log_dir='logs')
        tree.add_post_tick_handler(lambda t: board.track({
            'env_time': env.time,
            **env.render_info,
            agent_name: env.get_agent(agent_name).to_dict(),
        }))
        board.clear()
    return policy


def ppo_model_train():
    env = Dogfight2dEnv(options=options, render_mode='rgb_array')

    train_env = ModelGoToLocationTrainWrapper(env=env, policies=[
        create_bt_greedy_policy(
                env=env, agent_name=options.blue_agents[0], filepath='policies/follow_route.xml'
        ),
    ], agent_name=options.red_agents[0])

    model = PPO("MlpPolicy", train_env, verbose=2, tensorboard_log=f"./logs/{MODEL_NAME}")
    model.learn(total_timesteps=250000, progress_bar=True)
    model.save(MODEL_PATH)


def ppo_test():
    options = Options()
    options.delta_time = 0.5  # 每次更新的间隔
    options.self_side = 'red'
    options.simulation_rate = 100
    env = Dogfight2dEnv(options=options, render_mode='human')
    # model = PPO.load(f'./models/{MODEL_NAME}')

    policy = pydogfight.policy.MultiAgentPolicy(
            # pydogfight.policy.ModelPolicy(
            #         env=env, model=model,
            #         agent_name=options.red_agents[0],
            #         update_interval=options.policy_interval),
            create_bt_model_policy(
                    env=env, agent_name=options.red_agents[0],
            ),
            # ManualPolicy(env=env, control_agents=options.agents, delta_time=0),
            # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
            create_bt_greedy_policy(
                    env=env, agent_name=options.blue_agents[0], filepath='policies/follow_route.xml'
            ),
    )

    win_count = {
        'red' : 0,
        'blue': 0,
        'draw': 0
    }

    env.render_info['red_wins'] = win_count['red']
    env.render_info['blue_wins'] = win_count['blue']
    env.render_info['draw'] = win_count['draw']

    for _ in range(TEST_N):
        if not env.isopen:
            break
        obs, info = env.reset()
        policy.reset()

        while env.isopen:
            policy.select_action()
            policy.put_action()
            info = env.gen_info()
            env.gen_reward(color='red')
            env.gen_reward(color='blue')
            if info['terminated'] or info['truncated']:
                if info['winner'] in win_count:
                    win_count[info['winner']] += 1
                    env.render_info['red_wins'] = win_count['red']
                    env.render_info['blue_wins'] = win_count['blue']
                    env.render_info['draw'] = win_count['draw']
                break
            if env.should_update():
                env.update()
            if env.should_render():
                env.render()
        with open(f'logs/{MODEL_NAME}_test.json', 'a') as f:
            json.dump(env.render_info, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # ppo_model_train()
    # ppo_teacher_train()
    ppo_test()

    # options = Options()
    # options.delta_time = 1
    # options.simulation_rate = 1000
    # options.step_update_delta_time = 0
    # env = Dogfight2dEnv(options=options, render_mode='rgb_array')
    #
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("ppo_dogfight_2d")
    #
    # del model  # remove to demonstrate saving and loading
    #
    # model = PPO.load("ppo_dogfight_2d")
    # options = Options()
    # options.delta_time = 0.1
    # options.simulation_rate = 100
    # env = Dogfight2dEnv(options=options, render_mode='render')
    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, terminated, truncated, info = env.step(action)
    #     env.render()
