from pydogfight.policy.bt import *
from pydogfight.policy.bt.rl import *
import unittest
from pydogfight import Options, Dogfight2dEnv
from pydogfight.policy import Policy, ManualPolicy, MultiAgentPolicy, BTPolicy, DogfightTree


class TestRL(unittest.TestCase):

    def setUp(self):
        self.options = Options()
        self.env = Dogfight2dEnv(options=self.options, render_mode='rgb_array')
        self.env.reset()
        self.agent_name = self.options.red_agents[0]

    def test_rl_int_value(self):
        node = RLIntValue(key='int_value', high=10)
        tree = DogfightTree(
                env=self.env,
                agent_name=self.agent_name,
                root=node,
                name=self.agent_name,
                context={ }
        )
        tree.setup()
        policy = BTPolicy(
                env=self.env,
                tree=tree,
                agent_name=self.agent_name,
        )
        policy.reset()

        for i in range(10):
            policy.take_action()
            policy.put_action()
            print(tree.context['int_value'])
            self.assertGreaterEqual(tree.context['int_value'], 0)
            self.assertLessEqual(tree.context['int_value'], 9)
            self.env.update()

    def test_rl_float_value(self):
        node = RLFloatValue(key='float_value', high=10, algo='SAC')
        tree = DogfightTree(
                env=self.env,
                agent_name=self.agent_name,
                root=node,
                name=self.agent_name,
                context={ }
        )
        tree.setup()
        policy = BTPolicy(
                env=self.env,
                tree=tree,
                agent_name=self.agent_name,
        )
        policy.reset()

        for i in range(10):
            policy.take_action()
            policy.put_action()
            print(tree.context['float_value'])
            self.assertGreaterEqual(tree.context['float_value'], 0)
            self.assertLessEqual(tree.context['float_value'], 9)
            self.env.update()


