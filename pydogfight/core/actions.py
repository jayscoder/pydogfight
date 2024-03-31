# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum

class Actions(IntEnum):
    # Turn left, turn right, move forward
    keep = 0  # 什么也不做，保持原来的状态
    go_to_location = 1  # 飞到指定位置 (x, y)
    fire_missile = 2  # 朝目标点发射导弹 (x, y)
    go_home = 3 # 返航

