# from numpy import pi
# import matplotlib.pyplot as plt
# from pyclothoids import Clothoid
# clothoid0 = Clothoid.G1Hermite(0, 0, pi, 1, 1, 0)
# plt.plot( *clothoid0.SampleXY(500) )
# print(clothoid0.dk, clothoid0.KappaStart, clothoid0.KappaEnd)
# Use the parameter env_id to make the environment
# env = gym.make(env_id, render_mode='human')
import uuid

class A:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def creator(cls, d: dict):
        return cls(**d)


class B(A):

    def __init__(self, name: str, age: int):
        super().__init__(name=name)
        self.age = age


if __name__ == '__main__':
    id = uuid.uuid4()
    print(id.hex)
