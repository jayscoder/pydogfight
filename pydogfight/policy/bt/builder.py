import pybts.rl.builder

from pydogfight.policy.bt.base_class import *
from pydogfight.policy.bt.nodes_actions import *
from pydogfight.policy.bt.nodes_conditions import *
from pydogfight.policy.bt.nodes_pursue import *
from pydogfight.policy.bt.nodes_explore import *
from pydogfight.policy.bt.manual import ManualControl
from pydogfight.policy.bt.nodes_evade import *
from pydogfight.policy.bt.nodes_fire import *
from pydogfight.policy.bt.nodes_rl import *

import os

BASE_DIR = os.path.dirname(__file__)


class BTPolicyBuilder(pybts.rl.builder.RLBuilder):
    def register_default(self):
        super().register_default()

        self.register_node(
                IsMissileThreatDetected,
                IsEnemyDetected,
                GoToCenter,
                IsInSafeArea,
                IsOnActiveRoute,
                EvadeMissile,
                FireMissileAtNearestEnemy,
                GoToNearestEnemy,
                GoToSafeArea,
                Explore,
                GoHome,
                IsMissileDepleted,
                IsFuelBingo,
                FollowRoute,
                KeepFly,
                GoToLocation,
                IsReachLocation,
                ManualControl,
                IsWin,
                IsLose,
                IsDraw,
                IsNearEnemy,
                IsInGameRange,
                IsOutOfGameRange,
                IsFuelDepleted,
                IsMissileFull,
                IsFireMissileOverFreqLimit,
                RLFireAndGoToLocation,
                FireMissileAtNearestEnemyWithHitPointCheck,
                CanFireMissile
        )
        
        # Explore
        self.register_node(
                Explore,
                GridExplore
        )

        # pursue
        self.register_node(
                AwayFromNearestEnemy,
                PursueNearestEnemy,
                LeadPursueNearestEnemy,
                FPolePursueNearestEnemy,
                GoToNearestEnemy,
                PurePursueNearestEnemy,
                AutoPursueNearestEnemy,
                LagPursueNearestEnemy
        )

        # 强化学习节点
        self.register_node(
                RLSwitcher,
                ReactiveRLSwitcher,
                RLSelector,
                RLCondition,
                RLAction,
                RLActionPPA,
                RLIntValue,
                RLFloatValue,
                RLGoToLocation,
                RLFireAndGoToLocation
        )
