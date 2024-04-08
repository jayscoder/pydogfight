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
        self.register_bt(
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

        self.register_bt(
                BTPPOGoToLocationModel,
                PPOSwitcher,
                ReactivePPOSwitcher,
                PPOSelector,
                PPOCondition,
                PPOAction,
                PPOActionPPA,
        )

        # Ricky注册



if __name__ == '__main__':
    # _main()
    builder = BTPolicyBuilder()
    with open(os.path.join(BASE_DIR, 'bt_policy_nodes.md'), 'w') as f:
        texts = ['## 战机策略行为树节点定义']
        for k, n in builder.repo_node.items():
            texts.append(f'**{k}**')
            for line in n.__doc__.split('\n'):
                line = line.strip()
                if line == '':
                    continue
                texts.append(line.strip())

        f.write('\n\n'.join(texts))
