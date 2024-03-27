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

    def view_info(self):
        return {
            'age': self.age
        }

builder = pybt.xml_builder.BTXMLBuilder()
builder.register('Person', Person.creator)
root = builder.build_from_file('viewer_demo_bt.xml')
tree = pybt.node.BehaviourTree(root=root)

viewer = pybt.viewer.BTViewer(tree=tree, title='测试树')

if __name__ == '__main__':
    # print(pybt.utility.bt_to_echarts_json(root))
    viewer.run(asynchronous=False)
