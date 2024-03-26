from __future__ import annotations
import gymnasium as gym
from gymnasium.envs.registration import register
from .envs import Dogfight2dEnv
from .core.options import Options
from .core.constants import *
from .core.actions import *
from .core.world_obj import *
from .policy import *
from .wrappers import *
envs = [
    'Dogfight-2d-v0'
]

register(id='Dogfight-2d-v0', entry_point='gym_dogfight.envs:Dogfight2dEnv')
