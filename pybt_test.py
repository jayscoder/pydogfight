import time

from py_trees import common

import pybt


class Person(pybt.node.BTNode):
    def __init__(self, name: str, age: int):
        super().__init__(name=name)
        self.age = age

    @classmethod
    def creator(cls, d, c):
        return Person(name=d['name'], age=int(d['age']))

    def update(self) -> common.Status:
        return common.Status.SUCCESS

    def to_data(self):
        return {
            'age': self.age
        }


builder = pybt.builder.BTBuilder()
builder.register('Person', Person.creator)
root = builder.build_from_file('viewer_demo_bt.xml')
tree = pybt.BTTree(root=root, name='Person')

bt_board = pybt.board.BTBoard(tree=tree, log_dir='logs')

if __name__ == '__main__':
    bt_board.clear()
    for i in range(10000):
        tree.tick()
        bt_board.track()
        time.sleep(0.5)
        print(i)
