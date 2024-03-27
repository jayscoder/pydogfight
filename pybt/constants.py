import py_trees
from py_trees.common import Status


def _hex_color(color: str) -> tuple[int, int, int]:
    # 将十六进制颜色代码转换为RGB值
    r = int(color[0:2], 16)  # 转换红色分量
    g = int(color[2:4], 16)  # 转换绿色分量
    b = int(color[4:6], 16)  # 转换蓝色分量
    return (r, g, b)


def _echarts_color(color: str) -> str:
    r, g, b = _hex_color(color)
    return f'rgb({r},{g},{b})'


# Map of color names to RGB values
ECHARTS_COLORS = {
    "red"   : _echarts_color('d81e06'),
    "blue"  : _echarts_color('1296db'),
    "green" : _echarts_color('1afa29'),
    "pink"  : _echarts_color('d4237a'),
    "yellow": _echarts_color('f4ea2a'),
    "grey"  : _echarts_color('8a8a8a'),
    'black' : _echarts_color('000000'),
    'white' : _echarts_color('ffffff')
}

STATUS_TO_ECHARTS_COLORS = {
    Status.SUCCESS: ECHARTS_COLORS['green'],
    Status.INVALID: ECHARTS_COLORS['grey'],
    Status.RUNNING: ECHARTS_COLORS['blue'],
    Status.FAILURE: ECHARTS_COLORS['red']
}



class ECHARTS_SYMBOLS:
    CIRCLE = 'circle'
    RECT = 'rect'
    ROUNDRECT = 'roundRect'
    TRIANGLE = 'triangle'
    DIAMOND = 'diamond'
    PIN = 'pin'
    ARROW = 'arrow'
    NONE = 'none'

