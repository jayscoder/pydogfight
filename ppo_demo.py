from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gym_dogfight import *
import json


def ppo_model_train():
    options = Options()
    options.delta_time = 1  # 每次更新的间隔
    options.self_side = 'red'
    options.simulation_rate = 1000
    options.step_update = 1
    env = Dogfight2dEnv(options=options, render_mode='rgb_array')

    train_env = ModelTrainWrapper(env=env, policies=[
        GreedyPolicy(env=env, agent_name=options.blue_agents[0], delta_time=5)
    ], agent_name=options.red_agents[0])

    model = PPO("MlpPolicy", train_env, verbose=2, tensorboard_log="./logs/ppo_model_train/")
    model.learn(total_timesteps=250000, progress_bar=True)
    model.save("./models/ppo_model_train")


def ppo_teacher_train():
    options = Options()
    options.delta_time = 1  # 每次更新的间隔
    options.self_side = 'red'
    options.simulation_rate = 1000
    # options.step_update = 1
    env = Dogfight2dEnv(options=options, render_mode='rgb_array')

    train_env = TeacherTrainWrapper(
            env=env,
            policies=[
                GreedyPolicy(env=env, agent_name=options.blue_agents[0], delta_time=5)
            ], agent_name=options.red_agents[0],
            teacher=GreedyPolicy(env=env, agent_name=options.red_agents[0], delta_time=1))

    model = PPO("MlpPolicy", train_env, verbose=2, tensorboard_log="./logs/ppo_teacher_train/")
    model.learn(total_timesteps=250000, progress_bar=True)
    model.save("./models/ppo_teacher_train")


def ppo_test(model_name: str = 'ppo_model_train'):
    options = Options()
    options.delta_time = 0.5  # 每次更新的间隔
    options.self_side = 'red'
    options.simulation_rate = 100
    env = Dogfight2dEnv(options=options, render_mode='human')
    model = PPO.load(f'./models/{model_name}')

    policy = MultiAgentPolicy(
            env=env,
            policies=[
                ModelPolicy(env=env, model=model, agent_name=options.red_agents[0], delta_time=1),
                # ManualPolicy(env=env, control_agents=options.agents, delta_time=0),
                # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
                GreedyPolicy(env=env, agent_name=options.blue_agents[0], delta_time=5)
            ],
    )

    win_count = {
        'red' : 0,
        'blue': 0,
        'draw': 0
    }

    env.render_info['red_wins'] = win_count['red']
    env.render_info['blue_wins'] = win_count['blue']
    env.render_info['draw'] = win_count['draw']

    obs, info = env.reset()
    i = 0
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
            obs, info = env.reset()
            policy.reset()
        if env.should_update():
            env.update()
        if env.should_render():
            env.render()
        i += 1
        if i % 100 == 0:
            with open('logs/ppo_log.json', 'a') as f:
                json.dump(env.render_info, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # ppo_model_train()
    # ppo_teacher_train()
    ppo_test('ppo_teacher_train')
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
