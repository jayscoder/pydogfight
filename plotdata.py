import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

class Ploter():

    def __init__(self) -> None:
        self.x_start = None
        self.x_end = None
        self.smooth_len = None
        self.min_data_len = -1

    def reset(self):
        self.min_data_len = -1

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

    def plot_csv(self, filepath:str):
        data = pd.read_csv(filepath)
        steps = data['Step']
        values = data['Value']
        if self.smooth_len is not None:
            values = self.smooth(values)

        filename = os.path.basename(filepath)
        label = filename.split('.')[0]
        self.min_data_len = min(self.min_data_len, len(steps)) if self.min_data_len != -1 else len(steps)
        plt.plot(steps, values, label=label)

    def plot_csv_from_dir(self, dirpath:str, title:str=None, xlabel='Step', ylabel='Value', show=True, x_cut=True):
        """
        dirpath:文件夹路径
        title:生成图标题
        show:是否展示图片
        x_cut:是否开启x轴截断
        """
        self.reset()
        # 获取全部csv文件
        csv_files = glob.glob(dirpath+'/*.csv')
        
        for file in csv_files:
            self.plot_csv(file)

        dirname = os.path.basename(dirpath)
        plt.title(dirname if not title else title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        if x_cut:
            plt.xlim(self.x_start if self.x_start is not None else 20,
                    self.x_end if self.x_end is not None else self.min_data_len)

        plt.savefig(dirpath+'/'+dirname+'.png')
        if show:
            plt.show()
        

pt = Ploter()
pt.set_cut_range(50,2000)
pt.set_smooth_len(10)
folder_path = "data"
items = os.listdir(folder_path)
for item in items:
    item_path = os.path.join(folder_path, item)
    if os.path.isdir(item_path):
        pt.plot_csv_from_dir(item_path,show=True,x_cut=True)
# pt.plot_csv_from_dir(folder_path)
