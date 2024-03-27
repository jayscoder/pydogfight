from __future__ import annotations

from pybt.node import BTNode
import py_trees
import json
from pybt.constants import STATUS_TO_ECHARTS_COLORS, ECHARTS_SYMBOLS
import yaml


def bt_to_view_info(node: py_trees.behaviour.Behaviour) -> dict:
    info = {
        'id'      : node.id.hex,
        'name'    : node.name,
        'status'  : node.status.name,
        'children': len(node.children),
        # 'feedback_message': node.feedback_message,
        # 'qualified_name'  : node.qualified_name,
    }
    if isinstance(node, BTNode):
        info = {
            **info,
            **node.view_info()
        }
    return info


def bt_to_echarts_json(node: py_trees.behaviour.Behaviour, ignore_children: bool = False) -> dict:
    symbol = ECHARTS_SYMBOLS.RECT
    symbolSize = 20
    if isinstance(node, py_trees.composites.Composite):
        symbol = ECHARTS_SYMBOLS.DIAMOND
        symbolSize = 30
    elif isinstance(node, py_trees.decorators.Decorator):
        symbol = ECHARTS_SYMBOLS.CIRCLE
        symbolSize = 15

    info = bt_to_view_info(node)
    tooltip = yaml.dump(info, allow_unicode=True).replace('\n', '<br/>')

    d = {
        'name'      : node.id.hex,
        'value'     : node.name,
        'data'      : {
            'name'   : node.name,
            'id'     : node.id,
            'tooltip': tooltip
        },
        'itemStyle' : {
            'color'      : STATUS_TO_ECHARTS_COLORS[node.status],
            'borderColor': STATUS_TO_ECHARTS_COLORS[node.status],
        },
        'symbolSize': symbolSize,
        'symbol'    : symbol,
    }

    if not ignore_children:
        d['children'] = [bt_to_echarts_json(child, ignore_children) for child in node.children]
    else:
        d['children'] = []
    return d
