import pybts.rl.builder

from pydogfight.policy.bt.nodes_actions import *
from pydogfight.policy.bt.nodes_conditions import *
from pydogfight.policy.bt.nodes import *
from pydogfight.policy.bt.manual import ManualControl
from pydogfight.policy.bt.rl import *
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
                AttackNearestEnemy,
                GoToNearestEnemy,
                PursueNearestEnemy,
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
                AwayFromNearestEnemy,
                IsInGameRange,
                IsOutOfGameRange,
                IsFuelDepleted,
                IsMissileFull
        )

        # 强化学习节点
        self.register_node(
                BTPPOGoToLocationModel,
                PPOSwitcher,
                ReactivePPOSwitcher,
                PPOSelector,
                PPOCondition,
                PPOAction,
                PPOActionPPA,
        )

        # Ricky注册
