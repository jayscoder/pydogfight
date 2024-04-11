from __future__ import annotations

import typing

from pybts.rl import RLTree
from pydogfight.envs.dogfight_2d_env import Dogfight2dEnv
from pybts import Node


class DogfightTree(RLTree):

    def __init__(self, env: Dogfight2dEnv, root: Node, name: str, agent_name: str, context: dict = None) -> None:
        if agent_name in env.options.red_agents:
            agent_color = 'red'
        else:
            agent_color = 'blue'
        if context is None:
            context = { }

        context.update({
            'agent_name' : agent_name,
            'agent_color': agent_color,
            'time'       : env.time,
            'env'        : env,
            'agent'      : env.get_agent(agent_name).to_dict(),
            'options'    : env.options.to_dict()
        })
        super().__init__(root=root, name=name, context=context)
        self.env = env
        self.agent_name = agent_name
    
    def tick(
            self,
            pre_tick_handler: typing.Optional[
                typing.Callable[[DogfightTree], None]
            ] = None,
            post_tick_handler: typing.Optional[
                typing.Callable[[DogfightTree], None]
            ] = None,
    ) -> None:
        # 在tick之前更新时间、agent信息，方便后面使用
        self.context['time'] = self.env.time
        self.context['agent'] = self.env.get_agent(self.agent_name).to_dict()
        super().tick(pre_tick_handler=pre_tick_handler, post_tick_handler=post_tick_handler)
