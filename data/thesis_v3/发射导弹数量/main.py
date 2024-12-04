import pandas as pd


def treat(file: str):
    data = pd.read_csv(file)
    steps = data['Step']
    values = data['Value']

    # 计算平均值
    mean = values.mean()
    print(file, mean)


if __name__ == '__main__':
    treat('./ppo-E_2_red.csv')
    treat('./ppo-E_2_blue.csv')
    treat('./ppo-I_2_red.csv')
    treat('./ppo-I_2_blue.csv')
