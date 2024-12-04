import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Ploter:

    def __init__(self) -> None:
        self.x_start = None
        self.x_end = None
        self.smooth_len = None
        self.min_data_len = -1

    def reset(self):
        self.min_data_len = -1
        plt.clf()

    def set_cut_range(self, x_start, x_end):
        """
        设置x轴截断范围，如果某个值是None则起始默认20，终止默认为所有数据长度的最小值
        """
        self.x_start = x_start
        self.x_end = x_end

    def set_smooth_len(self, smooth_len):
        self.smooth_len = smooth_len
        self.x_start = smooth_len

    def smooth(self, data):
        smoothed_data = data.rolling(window=self.smooth_len, min_periods=1).mean()
        return smoothed_data

    def plot_csv(self, filepath: str):
        data = pd.read_csv(filepath)
        steps = data['Step']
        values = data['Value']
        if self.smooth_len is not None:
            values = self.smooth(values)

        filename = os.path.basename(filepath)
        label = filename.split('.')[0]
        self.min_data_len = min(self.min_data_len, len(steps)) if self.min_data_len != -1 else len(steps)
        plt.plot(steps, values, label=label)

        # std_dev = values.std()
        # plt.axhline(y=std_dev, color='r', linestyle='--', label=f'{label} Std Dev: {std_dev:.2f}')

    def plot_csv_from_dir(self, dirpath: str, title: str = None, xlabel='Step', ylabel='Value', show=True, x_cut=True):
        """
        dirpath:文件夹路径
        title:生成图标题
        show:是否展示图片
        x_cut:是否开启x轴截断
        """
        self.reset()
        # 获取全部csv文件
        csv_files = glob.glob(dirpath + '/*.csv')

        sorted_csv_files = sorted(csv_files, key=lambda x: x.split('/')[-1].split('.')[0])

        for file in sorted_csv_files:
            self.plot_csv(file)

        dirname = os.path.basename(dirpath)
        plt.title(dirname if not title else title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
        plt.grid(True)

        if x_cut:
            plt.xlim(self.x_start if self.x_start is not None else 20,
                     self.x_end if self.x_end is not None else self.min_data_len)

        plt.savefig(dirpath + '/' + dirname + '.png')
        if show:
            plt.show()


# if __name__ == '__main__':
#     folder_path = "."
#     items = os.listdir(folder_path)
#     for item in items:
#         item_path = os.path.join(folder_path, item)
#         if os.path.isdir(item_path):
#             print(item_path)
#             pt = Ploter()
#             pt.set_cut_range(50, 3000)
#             if '奖励' in item_path:
#                 pt.set_smooth_len(50)
#             pt.plot_csv_from_dir(item_path, show=True, x_cut=True)

def main():
    pt = Ploter()
    pt.set_cut_range(50, 3000)
    pt.plot_csv_from_dir('50轮胜率', show=False, x_cut=True, xlabel='Episode', ylabel='Win Rate')

    pt = Ploter()
    pt.set_cut_range(50, 3000)
    pt.set_smooth_len(50)
    pt.plot_csv_from_dir('50轮平均奖励', show=False, x_cut=True, xlabel='Episode', ylabel='Reward')


if __name__ == '__main__':
    main()

# pt.plot_csv_from_dir(folder_path)
