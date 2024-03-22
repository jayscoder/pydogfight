from __future__ import annotations
import gymnasium as gym
from gymnasium.envs.registration import register
from .envs import Dogfight2dEnv
from .core.options import Options

envs = [
    'Dogfight-2d-v0'
]

register(id='Dogfight-2d-v0', entry_point='gym_dogfight.envs:Dogfight2dEnv')
