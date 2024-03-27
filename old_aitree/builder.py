from __future__ import annotations
import xml.etree.ElementTree as ET
from .behaviour import Behaviour
import os
from . import utilities


def build_from_xml(xml_text, ignore_children=False) -> Behaviour:
    """
    从xml字符串构建节点
    :param xml_text: xml字符串
    :param ignore_children: 是否忽略子树
    :return:
    """
    xml_root = ET.fromstring(xml_text)
    return build_from_xml_node(xml_root, ignore_children=ignore_children)


def build_from_xml_node(xml_node: ET.Element, ignore_children=False) -> Behaviour:
    """
    从xml节点构建节点
    :param xml_node: xml节点
    :param ignore_children: 是否忽略子树
    :return:
    """
    from .register import find_node_cls
    node_cls = find_node_cls(xml_node.tag)
    if node_cls is None:
        raise Exception(f"node class {xml_node.tag} not found")
    node = node_cls(**xml_node.attrib)

    if not ignore_children:
        for child in xml_node:
            node.add_child(build_from_xml_node(child, ignore_children=ignore_children))
    return node


def build_from_xml_file(filepath: str) -> Behaviour:
    if not os.path.exists(filepath):
        raise Exception(f'path: {filepath} not exists')

    files = []
    if os.path.isdir(filepath):
        files.extend(utilities.extract_xml_files_from_dir(filepath))
    else:
        files.append(filepath)

    if len(files) == 1:
        xml_root = ET.parse(files[0]).getroot()
        node = build_from_xml_node(xml_root, ignore_children=False)
        node.filepath = files[0]
    else:
        node = Behaviour()
        node.filepath = filepath
        for path in files:
            child = build_from_xml_file(path)
            node.add_child(child)
    return node
