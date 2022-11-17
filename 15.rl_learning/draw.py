import sys

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./result_is_profit/stock_profit_all.csv")
df = df.drop(columns=['date', 'code', 'open', 'high', 'low', 'close', 'isST', 'tradestatus', 'pctChg','adjustflag'])

x_num = 4  # 画板横向几个图
y_num = 6  # 画板纵向几个图


def draw(df):
    i = 0
    plt.figure(figsize=(20, 10), dpi=300)  # 画板大小
    for column in df.columns.values:
        i = i + 1
        values = df[column].values
        plt.subplot(y_num, x_num, i)  # 表示第i张图片，下标只能从1开始，不能从0，
        plt.title(label=column)
        plt.hist(values, bins=100)
    plt.grid()  # 网格
    plt.show()


# 全部
draw(df)
# 正收益
df_active = df[df["profit"] > 0]
draw(df_active)
# 负收益
df_nagtive = df[df["profit"] < 0]
draw(df_nagtive)

sys.exit()
