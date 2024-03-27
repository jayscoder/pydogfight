#!/usr/bin/env python
#
# License: BSD
#   https://raw.githubusercontent.com/splintered-reality/py_trees/devel/LICENSE
#
##############################################################################
# Documentation
##############################################################################

"""Assorted utility functions."""

##############################################################################
# Imports
##############################################################################

from __future__ import annotations

import multiprocessing
import os
import re
import traceback
import typing

##############################################################################
# Python Helpers
##############################################################################

C = typing.TypeVar("C", bound=typing.Callable)


# TODO: This currently doesn't work well with mypy - dynamic typing
# is not its thing. Need to find a way to make this work without
# creating errors on the user side. In the docstring's example, usage
# of the static 'counter' variable results in:
#
# error: "Callable[[], Any]" has no attribute "counter"  [attr-defined]


def static_variables(**kwargs: typing.Any) -> typing.Callable[[C], C]:
    """
    Attach initialised static variables to a python method.

    .. code-block:: python

       @static_variables(counter=0)
       def foo():
           foo.counter += 1
           print("Counter: {}".format(foo.counter))
    """

    def decorate(func: C) -> C:
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_variables(primitives={ bool, str, int, float })
def is_primitive(incoming: typing.Any) -> bool:
    """
    Check if an incoming argument is a primitive type with no esoteric accessors.

    That is, it has no class attributes or container style [] accessors.

    Args:
        incoming: the instance to check
    Returns:
        True or false, depending on the check against the reserved primitives
    """
    return type(incoming) in is_primitive.primitives  # type: ignore[attr-defined]


def truncate(original: str, length: int) -> str:
    """
    Provide an elided (...) version of a string if it is longer than desired.

    Args:
        original: string to elide
        length: constrain the elided string to this
    """
    s = (original[: length - 3] + "...") if len(original) > length else original
    return s


##############################################################################
# System Tools
##############################################################################


class Process(multiprocessing.Process):
    """Convenience wrapper around multiprocessing.Process."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self) -> None:
        """Start the process, handle exceptions if needed."""
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self) -> typing.Any:
        """
        Check the connection, if there is an error, reflect it as an exception.

        Returns:
            The exception.
        """
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def which(program: str) -> typing.Optional[str]:
    """
    Call the command line 'which' tool (convenience wrapper).

    Args:
        program: name of the program to find.

    Returns:
        path to the program or None if it doesnt exist.
    """

    def is_exe(fpath: str) -> bool:
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, unused_fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def get_valid_filename(s: str) -> str:
    """
    Clean up and style a string so that it can be used as a filename.

    This is valid only from the perspective of the py_trees package. It does
    place a few extra constraints on strings to keep string handling and
    manipulation complexities to a minimum so that sanity prevails.

    * Removes leading and trailing spaces
    * Convert other spaces and newlines to underscores
    * Remove anything that is not an alphanumeric, dash, underscore, or dot

    .. code-block:: python

        >>> utilities.get_valid_filename("john's portrait in 2004.jpg")
        'johns_portrait_in_2004.jpg'

    Args:
        program (:obj:`str`): string to convert to a valid filename

    Returns:
        :obj:`str`: a representation of the specified string as a valid filename
    """
    s = str(s).strip().lower().replace(" ", "_").replace("\n", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def get_fully_qualified_name(instance: object) -> str:
    """
    Retrieve the fully qualified name of an object.

    For example, an instance of
    :class:`~py_trees.composites.Sequence` becomes 'py_trees.composites.Sequence'.

    Args:
        instance (:obj:`object`): an instance of any class

    Returns:
        :obj:`str`: the fully qualified name
    """
    module = instance.__class__.__module__
    # if there is no module, it will report builtin, get that
    # string via what should remain constant, the 'str' class
    # and check against that.
    builtin = str.__class__.__module__
    if module is None or module == builtin:
        return instance.__class__.__name__
    else:
        return module + "." + instance.__class__.__name__


import uuid
from typing import Union, Tuple
import xml.etree.ElementTree as ET
import os
import json


def new_node_id():
    # 只取uuid的前10位
    return uuid.uuid4().hex[:10]


def camel_case_to_snake_case(name):
    """
    驼峰转蛇形
    :param name:
    :return:
    """
    return ''.join(['_' + i.lower() if i.isupper() else i for i in name]).lstrip('_')


def parse_prop_options(options: Union[list, dict, str]) -> list:
    """
    解析属性选项
    :param options:
    :return:
    """
    if isinstance(options, str):
        options = options.split(',')
    if isinstance(options, list):
        result = []
        for option in options:
            if isinstance(option, str):
                result.append({
                    'name' : option,
                    'value': option
                })
            elif isinstance(option, dict):
                result.append({
                    'name' : option.get('name', ''),
                    'value': option.get('value', '')
                })
        return result
    elif isinstance(options, dict):
        result = []
        for key in options:
            result.append({
                'name' : key,
                'value': options[key]
            })
        return result
    return []


PROP_TYPE_MAPPER = {
    'str'   : str,
    'string': str,
    'int'   : int,
    'float' : float,
    'double': float,
    'number': float,
    'bool'  : bool,
    'list'  : list,
    'dict'  : dict,
    'json'  : dict,
}


def parse_prop_type(prop_type: [str, type]):
    if isinstance(prop_type, str):
        prop_type = prop_type.lower()
        if prop_type in PROP_TYPE_MAPPER:
            return PROP_TYPE_MAPPER[prop_type]
        else:
            return str
    else:
        return prop_type


def parse_type_value(value, value_type):
    value_type = parse_prop_type(value_type)
    if value_type == bool:
        return parse_bool_value(value)
    elif value_type == int:
        return parse_int_value(value)
    elif value_type == float:
        return parse_float_value(value)
    elif value_type == str:
        return str(value)
    elif value_type == list:
        return parse_list_value(value)
    elif value_type == dict:
        return parse_dict_value(value)
    elif callable(value_type):
        return value_type(value)
    return value


# 最终props都是以列表的形式保存的
def parse_props(props):
    if props is None:
        return []
    result = []
    if isinstance(props, list):
        for prop in props:
            if isinstance(prop, str):
                result.append({
                    'name'    : prop,
                    'type'    : 'str',
                    'default' : '',
                    'required': False,
                    'desc'    : '',
                    'options' : None,  # 选项 用于下拉框 仅在type为str时有效 {'name': '选项1', 'value': '1'}
                    'visible' : True,  # 是否可见
                })
            elif isinstance(prop, dict):
                result.append({
                    'name'    : prop.get('name', ''),
                    'type'    : prop.get('type', 'str'),
                    'default' : prop.get('default', ''),
                    'required': prop.get('required', False),
                    'desc'    : prop.get('desc', ''),
                    'options' : prop.get('options', None),
                    'visible' : prop.get('visible', True),
                })
    elif isinstance(props, dict):
        for prop in props:
            prop_item = props[prop]
            if isinstance(prop_item, dict):
                result.append({
                    'name'    : prop,
                    'type'    : prop_item.get('type', 'str'),
                    'default' : prop_item.get('default', ''),
                    'required': prop_item.get('required', False),
                    'desc'    : prop_item.get('desc', ''),
                    'options' : prop_item.get('options', None),
                    'visible' : prop_item.get('visible', True),
                })
            elif isinstance(prop_item, type):
                result.append({
                    'name'    : prop,
                    'type'    : prop_item,
                    'default' : '',
                    'required': False,
                    'desc'    : '',
                    'options' : None,
                    'visible' : True,
                })

    for i, item in enumerate(result):
        result[i]['type'] = parse_prop_type(item['type']).__name__
        if not callable(item['default']):
            result[i]['default'] = parse_type_value(value=item['default'], value_type=item['type'])
        result[i]['options'] = parse_prop_options(item['options'])

    return result


def merge_props(props: list, to_props: list):
    """
    合并两个props
    :param props:
    :param to_props:
    :return:
    """
    if to_props is None:
        return props
    to_props = to_props.copy()
    for prop in props:
        find_index = find_prop_index(to_props, prop['name'])
        if find_index == -1:
            to_props.append(prop)
        else:
            to_props[find_index] = prop
    return to_props


def find_prop(meta, name):
    if 'props' in meta:
        for prop in meta['props']:
            if prop['name'] == name:
                return prop
    return None


def find_prop_index(props, name):
    for index, prop in enumerate(props):
        if prop['name'] == name:
            return index
    return -1


def parse_bool_value(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return value > 0
    elif isinstance(value, float):
        return value > 0.0
    elif isinstance(value, str):
        return value.lower() in ['true', '1', 'yes', 'y']
    return False


def parse_int_value(value: str, default: int = 0):
    try:
        return int(value)
    except:
        return default


def parse_float_value(value: str, default: float = 0.0):
    try:
        return float(value)
    except:
        return default


def parse_list_value(value: str, default: list = None):
    try:
        value = json.loads(value)
        return value
    except:
        return default


def parse_dict_value(value: str, default: dict = None):
    try:
        value = json.loads(value)
        return value
    except:
        return default


# 定义一个函数将 XML 元素转换为字典
def xml_to_dict(element):
    result = {
        'tag'       : element.tag,
        'attributes': element.attrib,
        'children'  : [xml_to_dict(child) for child in element]
    }
    return result


def read_xml_to_dict(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return xml_to_dict(root)


# 从目录中提取出所有的xml文件
def extract_xml_files_from_dir(dir_path: str):
    xml_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    return xml_files
