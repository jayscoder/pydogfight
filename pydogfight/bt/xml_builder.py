from __future__ import annotations

from typing import Callable, Tuple, List
from py_trees.behaviour import Behaviour
from py_trees.composites import *
from py_trees.decorators import *
import xml.etree.ElementTree as ET
import copy
import py_trees


class BTXMLBuilder:
    def __init__(self):
        self.repo = { }
        self.register_default()

    def register(self, name: str | list[str], creator: Callable[[dict, [Behaviour]], Behaviour]):
        if isinstance(name, str):
            self.repo[name] = creator
        else:
            for _name in name:
                self.register(_name, creator)

    def build_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            tree = ET.parse(f)
            root = tree.getroot()
        return self.build_from_xml_node(root)

    def build_from_xml_text(self, xml: str, ignore_children: bool = False):
        xml_node = ET.fromstring(xml)
        return self.build_from_xml_node(xml_node=xml_node, ignore_children=ignore_children)

    def build_from_xml_node(self, xml_node: ET.Element, ignore_children: bool = False) -> Behaviour:
        assert xml_node.tag in self.repo
        creator = self.repo[xml_node.tag]
        children = []
        if not ignore_children:
            children = [self.build_from_xml_node(xml_node=child, ignore_children=ignore_children) for child in xml_node]
        data = copy.copy(xml_node.attrib)
        data['tag'] = xml_node.tag
        if 'name' not in data:
            data['name'] = xml_node.tag
        node: Behaviour = creator(data, children)
        return node

    def register_default(self):
        self.register('Sequence', lambda d, c: Sequence(name=d['name'], memory=d.get('memory', True), children=c))

        def c_parallel(d, c):
            policy = d.get('policy', 'SuccessOnOne')
            synchronise = d.get('synchronise', False)
            if policy == 'SuccessOnOne':
                return Parallel(name=d['name'],
                                policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=c)
            elif policy == 'SuccessOnAll':
                return Parallel(name=d['name'],
                                policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=synchronise), children=c)

        self.register('Parallel', c_parallel)
        self.register('Selector', lambda d, c: Selector(name=d['name'], memory=d.get('memory', True), children=c))
        self.register('Inverter', lambda d, c: Inverter(name=d['name'], child=c[0]))

