from random import randrange

from flask import Flask, render_template, render_template_string, send_file, send_from_directory, jsonify, request

from pyecharts import options as opts
from pyecharts.charts import Bar
import os
from pybt.node import BTNode
from py_trees.trees import BehaviourTree
from pybt import utility
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')


class BTViewer:

    def __init__(self, tree: BehaviourTree, title: str = 'pybt', update_interval: int = 0.5):
        self.title = title
        self.tree = tree
        self.update_interval = update_interval  # 每隔0.5s刷新一次
        self.history = { }
        self.paused = False
        self.app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)

        # 注册路由
        self.app.add_url_rule("/", view_func=self.index)
        self.app.add_url_rule('/static/<path:path>', view_func=self._send_static)
        self.app.add_url_rule('/get_config', view_func=self._get_config, methods=['GET'])
        self.app.add_url_rule('/get_tree', view_func=self._get_tree, methods=['GET'])
        self.app.add_url_rule('/get_history_times', view_func=self._get_history_times, methods=['GET'])
        self.app.add_url_rule('/pause', view_func=self.pause, methods=['POST'])
        self.app.add_url_rule('/play', view_func=self.play, methods=['POST'])
        self.app.add_url_rule('/reset', view_func=self.reset, methods=['POST'])

    def run(self, host: str = 'localhost', port: int = 10000, debug: bool = True, asynchronous: bool = True):
        if asynchronous:
            flask_thread = threading.Thread(target=lambda: self.app.run(host=host, port=port, debug=debug))
            flask_thread.start()
            return flask_thread
        else:
            self.app.run(host=host, port=port, debug=debug)

    def index(self):
        return send_from_directory(TEMPLATES_DIR, 'index.html')

    def _send_static(self, path):
        return send_from_directory(STATIC_DIR, path)

    def _get_config(self):
        return jsonify({
            'title'          : self.title,
            'update_interval': self.update_interval,
            'step'           : self.tree.count,
            'paused'         : self.paused
        })

    def _get_tree(self):
        step = request.args.get('step')
        if step is None:
            step = self.snapshot()
        if step in self.history:
            tree = self.history[step]
        else:
            tree = { 'name': 'NotFound' }

        return jsonify({
            'tree': tree,
            'step': step
        })

    def _get_history_times(self):
        keys = self.history.keys()
        # keys按照从小到大排序
        keys = sorted(keys)
        return jsonify({
            'steps': keys
        })

    def snapshot(self):
        """
        Save the current state of the root
        :return:
        """
        step = self.tree.count
        self.history[step] = utility.bt_to_echarts_json(self.tree.root, ignore_children=False)
        return step

    def pause(self):
        self.paused = True
        return ''

    def play(self):
        self.paused = False
        return ''

    def reset(self):
        self.history = { }
        return ''
