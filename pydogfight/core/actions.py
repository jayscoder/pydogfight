# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum
from typing import Tuple


class Actions(IntEnum):
    # Turn left, turn right, move forward
    keep = 0  # 什么也不做，保持原来的状态
    go_to_location = 1  # 飞到指定位置 (x, y)
    fire_missile = 2  # 朝目标点发射导弹 (x, y)
    go_home = 3  # 返航

    @classmethod
    def extract_action_in_value_range(cls, value, value_range: Tuple[float, float],
                                      actions: list[Actions] = None) -> Actions:
        if actions is None:
            actions = [cls.keep, cls.go_to_location, cls.fire_missile, cls.go_home]

        assert value_range[0] < value_range[1]
        assert len(actions) > 0
        value_length = value_range[1] - value_range[0]
        value_step = value_length / len(actions)
        for i in range(0, len(actions)):
            if value_range[0] + value_step * i <= value <= value_range[0] + value_step * (i + 1):
                return actions[i]
        return actions[-1]

    @classmethod
    def build(cls, value: str | int | Actions) -> Actions:
        if isinstance(value, int):
            return ACTIONS_MAP[value]
        elif isinstance(value, str):
            return ACTIONS_MAP[value.strip().lower()]
        elif isinstance(value, Actions):
            return value
        raise Exception('Invalid action value')


ACTIONS_MAP = {
    'keep'                      : Actions.keep,
    'go_to_location'            : Actions.go_to_location,
    'go_home'                   : Actions.go_home,
    'fire_missile'              : Actions.fire_missile,
    Actions.keep.value          : Actions.keep,
    Actions.go_to_location.value: Actions.go_to_location,
    Actions.go_home.value       : Actions.go_home,
    Actions.fire_missile.value  : Actions.fire_missile
}

if __name__ == '__main__':
    print(Actions.extract_action_in_value_range(actions=None, value=-1, value_range=(-1, 1)))
    allow_actions = 'keep, go_to_location, fire_missile'.split(',')
    allow_actions = list(map(lambda x: ACTIONS_MAP[x.strip().lower()], allow_actions))
    print(allow_actions)
