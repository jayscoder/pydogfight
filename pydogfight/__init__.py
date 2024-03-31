from __future__ import annotations
from gymnasium.envs.registration import register
from . import core
from . import wrappers
from . import policy
from .core.options import Options
from .core.actions import Actions
from .core.constants import *
from .core.world_obj import *
from .envs import Dogfight2dEnv

DOGFIGHT_ENVS = [
    'Dogfight-2d-v0'
]
register(id='Dogfight-2d-v0', entry_point='gym_dogfight.envs:Dogfight2dEnv')

print('pydogfight __init__.py', __file__)
