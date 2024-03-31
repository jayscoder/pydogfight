from pydogfight.wrappers.agent_wrapper import AgentWrapper
from pydogfight.policy import *


class ModelTrainWrapper(AgentWrapper):
    """
    指定单个Agent强化学习训练的视角
    """

    def __init__(self, policies: List[Policy], env: Dogfight2dEnv, agent_name: str = ''):
        super().__init__(env=env, agent_name=agent_name)
        self.observation_space = env.observation_space
        self.policies = policies

    def step(self, action):
        assert isinstance(self.env, Dogfight2dEnv)
        for p in self.policies:
            p.select_action()
            p.put_action()
        agent = self.env.get_agent(self.agent_name)
        agent.put_action(action)
        self.env.update()
        obs = self.env.gen_agent_obs(agent_name=self.agent_name)
        info = self.env.gen_info()
        reward = self.env.gen_reward(color=agent.color)
        return obs, reward, info['terminated'], info['truncated'], info


class TeacherTrainWrapper(ModelTrainWrapper):
    def __init__(self, teacher: Policy, policies: List[Policy], env: Dogfight2dEnv, agent_name: str = ''):
        super().__init__(policies=policies, env=env, agent_name=agent_name)
        self.teacher = teacher

    def step(self, action):
        t_reward = self.teacher_reward(action)
        obs, reward, terminated, truncated, info = super().step(action)
        reward += t_reward
        return obs, reward, terminated, truncated, info

    def teacher_reward(self, action):
        self.teacher.select_action()
        reward = 0
        # 如果动作和老师的动作接近，则给出更高的奖励
        while not self.teacher.actions.empty():
            teacher_action = self.teacher.actions.get_nowait()
            if action[0] == teacher_action[0]:
                dis = ((teacher_action[1] - action[1]) ** 2 + (teacher_action[2] - action[2]) ** 2) ** 0.5
                reward += 10000 / dis
        return reward
