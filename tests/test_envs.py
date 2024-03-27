from __future__ import annotations

import time

import gymnasium as gym
import pytest
import pydogfight

def test_dogfight_env(env_id):
    # Use the parameter env_id to make the environment
    # env = gym.make(env_id, render_mode='human')
    env = gym.make(env_id, render_mode="human") # for visual debugging

    # reset env
    curr_seed = 0
    env.reset()
    for i in range(100):
        env.render()
