import os

import multiprocessing
import sys

import pandas as pd

"""
合并股票原始数据和收益数据，存入一个新的csv，这样方便通过excel人工分析
"""


def create_or_append_to_csv(df, file_path):
    # test_result_df.index.names = ['datetime']
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False, encoding='utf_8_sig')  # 判断一下file是否存在 > 存在：追加 / 不存在：保持
    else:
        df.to_csv(file_path, header=True, index=False, encoding='utf_8_sig')  # 判断一下file是否存在 > 存在：追加 / 不存在：保持


files_result = os.listdir("result")
files_train = os.listdir("stockdata/train")
profit_df = pd.read_csv("result_is_profit/result_profit.csv")
for index, row in profit_df.iterrows():
    for file_data in files_train:
        if str(int(row["stock"])) in file_data:
            # 追加到csv
            df = pd.read_csv("stockdata/train/" + file_data)
            df = df.tail(1)
            df['profit'] = row["profit"]
            create_or_append_to_csv(df, "result_is_profit/stock_profit_all.csv")
sys.exit()
