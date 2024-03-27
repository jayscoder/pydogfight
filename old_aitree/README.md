
## props

props主要用来实现节点参数的配置
有如下几种配置方式
```python
# 简单接收方式
props = ['name', 'age', 'sex', 'msg']
# 对象接收方式
props={
    'name': str,
    'age': int,
    'sex': str,
    'msg': str,
}

props={
    'name': {
        'type': str,
        'default': 'default_name',
        'required': True,
    },
    'age': {
        'type': int,
        'default': 18,
        'required': True,
    },
    'sex': {
        'type': str,
        'default': 'male',
        'required': True,
    },
    'msg': {
        'type': str,
        'default': 'default_msg',
        'required': False,
    },
}

# props列表完整接收的参数类型
[
    {
        'name': 'name',
        'type': str,
        'default': 'default_name',
        'required': True,
    },
    {
        'name': 'age',
        'type': 'int',
        'default': 18,
        'required': True,
    },
    {
        'name': 'sex',
        'type': 'str',
        'default': 'male',
        'required': True,
    },
    {
        'name': 'msg',
        'type': str,
        'default': 'default_msg',
        'required': False,
    },
]
```

type: 参数类型（可以直接用python内置类型，也可以用字符串表示）
- str
- int
- float
- bool
