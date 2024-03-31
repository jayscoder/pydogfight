from pydogfight.policy.behavior_tree.greedy_nodes import *
from pydogfight.policy.behavior_tree.nodes import *
from pydogfight.policy.behavior_tree.model_nodes import *


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
                PursueNearestEnemy,
                Explore,
                GoHome,
                IsMissileDepleted,
                IsFuelBingo,
                FollowRoute,
                KeepFly,
                GoToLocation,
                IsReachLocation,
                BTPPOGoToLocationModel
        )

    def build_default(self) -> pybts.Node:
        return self.build_from_file(DEFAULT_BT_GREEDY_POLICY_FILE)


def _main():
    builder = BTPolicyBuilder()
    tree = builder.build_from_file(DEFAULT_BT_GREEDY_POLICY_FILE)
    # print(pybts.utility.bt_to_xml(tree))
    with open(DEFAULT_BT_GREEDY_POLICY_FILE, 'r') as f:
        # print(pybts.utility.xml_to_json(f.read()))
        json_data = pybts.utility.xml_to_json(f.read())
        tree = builder.build_from_json(json_data=json_data)
        # print(tree)
        # print(pybts.utility.bt_to_xml(tree))
        for node in tree.iterate():
            print(node.name)


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
