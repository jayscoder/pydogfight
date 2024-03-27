from pydogfight.utils.position_memory import *
import time


def test_bench():
    from pydogfight.core.options import Options
    start_time = time.time()
    N = 10
    SEP = 10
    boundary = Options().safe_boundary
    for i in range(N):
        position_memory = PositionMemoryV2(boundary=boundary, sep=SEP)
        position_memory.reset()
        print(position_memory.pick_position())

    end_time = time.time()
    print(f'{end_time - start_time} Seconds') # V1 2s
