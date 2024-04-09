from pydogfight.policy.bt.greedy_nodes import *
from pydogfight.policy.bt.nodes import *
from pydogfight.policy.bt.manual import ManualControl
from pydogfight.policy.bt.rl import *
import os

BASE_DIR = os.path.dirname(__file__)


class BTPolicyBuilder(pybts.Builder):
    def __init__(self):
        super().__init__()
        self.register_greedy()

    def register_greedy(self):
        self.register_node(
                InitGreedyPolicy,
                MissileThreatDetected,
                EnemyDetected,
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
                IsDraw
        )

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
