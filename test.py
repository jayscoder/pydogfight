from __future__ import annotations
import os

projects = []
for dirpath, dirnames, filenames in os.walk('logs'):
    if 'pybts.json' in filenames:
        # print(dirpath, dirnames, filenames)
        # 移除 log_dir 部分，获取相对路径
        relative_path = os.path.relpath(dirpath, 'logs')
        projects.append(relative_path)

if __name__ == '__main__':
    print(projects)

