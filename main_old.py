from __future__ import annotations

import os.path
from typing import Any, Dict

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from pydogfight import *
from pydogfight.policy import Policy, ManualPolicy
from pydogfight.wrappers import AgentWrapper
import json
import pybts
import pydogfight
import argparse

parser = argparse.ArgumentParser(description="PPO Training and Testing")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'teacher'],
                    help='Mode to run the script in: train, test, or teacher')
parser.add_argument('--render_mode', type=str, default='rgb_array', choices=['rgb_array', 'human'],
                    help='Render Mode')
parser.add_argument('--track_name',
                    type=str,
                    default='main',
                    help='BT Track Name')
parser.add_argument('--model_name', type=str, default='ppo_model',
                    help='PPO Model Name')
parser.add_argument('--train_timesteps', type=int, default=300000,
                    help='Train timesteps')
args = parser.parse_args()
print('args', args)

options = Options()
# options.delta_time = 0.5  # 每次更新的间隔
options.self_side = 'red'
options.simulation_rate = 1000
options.train = args.mode == 'train'
# options.policy_interval = 0.5
MODEL_NAME = args.model_name
MODEL_PATH = os.path.join('models', MODEL_NAME)
TEST_N = 100
BT_BOARD_TRACK_NAME = args.track_name


class ModelTrainProgressCallback(BaseCallback):
    def __init__(self, check_freq):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Step number: {self.n_calls}")
            print(f"Game Info: {self.training_env.get_attr('game_info')}")
        return True


class ModelTrainWrapper(AgentWrapper):
    """
    指定单个Agent强化学习训练的视角
    """

    def __init__(self, policies: list[Policy], env: Dogfight2dEnv, agent_name: str = ''):
        super().__init__(env=env, agent_name=agent_name)
        self.observation_space = env.observation_space
        # 组合离散和连续动作空间
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.policies = policies

    def step(self, action):
        assert isinstance(self.env, Dogfight2dEnv)
        for p in self.policies:
            p.take_action()
            p.put_action()
        agent = self.env.get_agent(self.agent_name)
        relative_waypoint = agent.waypoint.relative_move(
                dx=action[1] * agent.radar_radius,
                dy=action[2] * agent.radar_radius)
        action_type = Actions.extract_action_in_value_range(
                value=action[0],
                value_range=(-1, 1)
        )
        agent.put_action((action_type, relative_waypoint.x, relative_waypoint.y))
        old_info = self.env.gen_info()
        self.env.update()
        obs = self.env.gen_agent_obs(agent_name=self.agent_name)
        info = self.env.gen_info()
        reward = self.env.gen_reward(color=agent.color, previous=old_info)
        return obs, reward, info['terminated'], info['truncated'], info


def create_bt_policy(env: Dogfight2dEnv, agent_name: str, filepath: str, track: bool):
    filename = os.path.basename(filepath).replace('.xml', '')
    tree = pybts.Tree(
            root=pydogfight.policy.BTPolicyBuilder().build_from_file(filepath),
            name=os.path.join(BT_BOARD_TRACK_NAME, agent_name, filename))
    policy = pydogfight.policy.BTPolicy(
            env=env,
            tree=tree,
            agent_name=agent_name,
            update_interval=options.policy_interval
    )

    if track:
        board = pybts.Board(tree=policy.tree, log_dir='logs')
        tree.add_post_tick_handler(lambda t: board.track({
            'env_time': env.time,
            **env.render_info,
            agent_name: env.get_agent(agent_name).to_dict(),
        }))
        board.clear()
    return policy


def create_bt_model_policy(env: Dogfight2dEnv, agent_name: str, track: bool):
    tree = pybts.Tree(
            root=pydogfight.policy.BTPPOModel(
                    model=MODEL_PATH
            ),
            name=str(os.path.join(BT_BOARD_TRACK_NAME, MODEL_NAME)))

    policy = pydogfight.policy.BTPolicy(
            env=env,
            tree=tree,
            agent_name=agent_name,
            update_interval=options.policy_interval
    )

    if track:
        board = pybts.Board(tree=policy.tree, log_dir='logs')
        tree.add_post_tick_handler(lambda t: board.track({
            'env_time': env.time,
            **env.render_info,
            agent_name: env.get_agent(agent_name).to_dict(),
        }))
        board.clear()
    return policy


def run_train():
    env = Dogfight2dEnv(options=options, render_mode=args.render_mode)

    train_env = ModelTrainWrapper(env=env, policies=[
        create_bt_policy(
                env=env, agent_name=options.blue_agents[0], filepath='policies/follow_route.xml',
                track=False
        ),
    ], agent_name=options.red_agents[0])

    model = PPO("MlpPolicy", train_env, verbose=2, tensorboard_log=f"./logs/{MODEL_NAME}")
    model.learn(total_timesteps=args.train_timesteps, progress_bar=True,
                callback=ModelTrainProgressCallback(check_freq=1000))
    model.save(MODEL_PATH)


def run_test():
    options = Options()
    options.delta_time = 0.5  # 每次更新的间隔
    options.self_side = 'red'
    options.simulation_rate = 100
    env = Dogfight2dEnv(options=options, render_mode=args.render_mode)
    # model = PPO.load(f'./models/{MODEL_NAME}')

    policy = pydogfight.policy.MultiAgentPolicy(
            # pydogfight.policy.ModelPolicy(
            #         env=env, model=model,
            #         agent_name=options.red_agents[0],
            #         update_interval=options.policy_interval),
            # create_bt_model_policy(
            #         env=env, agent_name=options.red_agents[0],
            #         track=True
            # ),
            # ManualPolicy(env=env, control_agents=options.agents, update_interval=0),
            # ManualPolicy(env=env, control_agents=options.blue_agents, delta_time=0.01),
            create_bt_policy(
                    env=env,
                    agent_name=options.red_agents[0],
                    filepath='policies/bt_greedy_red.xml',
                    track=True
            ),
            create_bt_policy(
                    env=env,
                    agent_name=options.blue_agents[0],
                    filepath='policies/bt_greedy_blue.xml',
                    track=True
            ),
    )

    env.render_info = {
        **env.render_info,
        **env.game_info
    }
    for _ in range(TEST_N):
        if not env.isopen:
            break
        env.reset()
        policy.reset()

        while env.isopen:
            policy.take_action()
            policy.put_action()
            info = env.gen_info()

            if info['terminated'] or info['truncated']:
                env.render_info = {
                    **env.render_info,
                    **env.game_info
                }
                break
            if env.should_update():
                env.update()
            if env.should_render():
                env.render()
        with open(f'logs/{MODEL_NAME}_test.json', 'a') as f:
            json.dump(env.render_info, f, ensure_ascii=False, indent=4)


def main():
    if args.mode == 'train':
        run_train()
    elif args.mode == 'test':
        run_test()


if __name__ == '__main__':
    main()
    # ppo_teacher_train()
    # ppo_test()

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
