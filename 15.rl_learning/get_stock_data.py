import sys
import traceback

import baostock as bs
import pandas as pd
import os
import multiprocessing
import talib
import numpy as np
import threading

OUTPUT = './stockdata'


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def stock2csv(df_stock: pd.DataFrame, path: str):
    # print("当前进程id：", os.getpid(), " 父进程id：", os.getppid())
    # print('当前线程id : %d' % threading.currentThread().ident)
    df_stock.to_csv(path, index=False)


class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start='1990-01-01',
                 date_end='2020-03-23',
                 processing_num_max=5000
                 ):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        self.processing_num_max = processing_num_max
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self):
        stock_df = self.get_codes_by_date(self.date_end)
        pool = multiprocessing.Pool(8)
        # 测试时：只拉去前100支股票
        # 股票计数器
        processing_num = 0
        for index, row in stock_df.iterrows():
            try:
                # 过滤股票名字带*号的，影响windows创建文件
                if '*' in row["code_name"]:
                    continue
                # 过滤指数
                if 'sz.399' in row["code"]:
                    continue
                # 过滤指数
                if 'sh.000' in row["code"]:
                    continue
                # 过滤北交所
                if 'bj.' in row["code"]:
                    continue
                # 过滤ST
                if 'ST' in row["code_name"]:
                    continue
                # 过滤可能退市的股票
                if '退' in row["code_name"]:
                    continue
                # 控制股票数：方便测试时候控制股票数量，避免拉取太多数据
                processing_num = processing_num + 1
                if processing_num >= self.processing_num_max:
                    break
                # 调用baostock获取股票数据，后续可以改成akshare获取数据
                print(f'processing {row["code"]} {row["code_name"]}')
                df_stock = bs.query_history_k_data_plus(row["code"], self.fields,
                                                        start_date=self.date_start,
                                                        end_date=self.date_end).get_data()
                # df_stock.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"].strip("*")}.csv', index=False)
                path = f'{self.output_dir}/{row["code"]}.{row["code_name"].strip("*")}.csv'
                # self.stock2csv(df_stock, path)
                # 转换数据类型为数值类型
                df_stock["open"] = pd.to_numeric(df_stock["open"], errors='coerce')
                df_stock["high"] = pd.to_numeric(df_stock["high"], errors='coerce')
                df_stock["low"] = pd.to_numeric(df_stock["low"], errors='coerce')
                df_stock["close"] = pd.to_numeric(df_stock["close"], errors='coerce')
                df_stock["volume"] = pd.to_numeric(df_stock["volume"], errors='coerce')
                df_stock["amount"] = pd.to_numeric(df_stock["amount"], errors='coerce')
                df_stock["adjustflag"] = pd.to_numeric(df_stock["adjustflag"], errors='coerce')
                df_stock["tradestatus"] = pd.to_numeric(df_stock["tradestatus"], errors='coerce')
                df_stock["pctChg"] = pd.to_numeric(df_stock["pctChg"], errors='coerce')
                df_stock["peTTM"] = pd.to_numeric(df_stock["peTTM"], errors='coerce')
                df_stock["pbMRQ"] = pd.to_numeric(df_stock["pbMRQ"], errors='coerce')
                df_stock["psTTM"] = pd.to_numeric(df_stock["psTTM"], errors='coerce')
                df_stock["pctChg"] = pd.to_numeric(df_stock["pctChg"], errors='coerce')
                # 数据扩展，自己利用talib计算指标并加入到原始数据中
                df_stock["ma5"] = talib.MA(df_stock['close'], timeperiod=5, matype=0)
                df_stock["ma10"] = talib.MA(df_stock['close'], timeperiod=10, matype=0)
                df_stock["ma20"] = talib.MA(df_stock['close'], timeperiod=20, matype=0)
                df_stock['ma5_0_signal'] = np.where(df_stock['close'] < df_stock['ma5'], -1, 1)
                df_stock['ma10_0_signal'] = np.where(df_stock['close'] < df_stock['ma10'], -1, 1)
                df_stock['ma20_0_signal'] = np.where(df_stock['close'] < df_stock['ma20'], -1, 1)
                df_stock['ma5_ma10_signal'] = np.where(df_stock['ma5'] < df_stock['ma10'], -1, 1)
                df_stock['ma5_ma20_signal'] = np.where(df_stock['ma5'] < df_stock['ma20'], -1, 1)
                # 计算k、d值
                df_stock['k'], df_stock['d'] = talib.STOCH(df_stock['high'], df_stock['low'], df_stock['close'], fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                df_stock['j'] = 3 * df_stock['k'] - 2 * df_stock['d']
                df_stock['kd_cross_signal'] = np.where(df_stock['k'] < df_stock['d'], -1, 1)
                # macd
                df_stock["macd"], df_stock["macdsignal"], df_stock["macdhist"] = talib.MACD(df_stock['close'], fastperiod=12, slowperiod=26, signalperiod=9)
                # 去除Nan值行
                df_stock.dropna(inplace=True)
                pool.apply_async(stock2csv, (df_stock, path,))
            except Exception as e:
                traceback.print_exc()
                pass
        pool.close()
        pool.join()
        self.exit()


def removeAllFiles(basePath):
    """
    删除目录下的所有文件，一层目录，不递归
    :param basePath:
    :return:
    """
    if os.path.exists(basePath):
        files = os.listdir(basePath)
        for file in files:
            if os.path.isfile(basePath + "/" + file):
                os.remove(basePath + "/" + file)


if __name__ == '__main__':
    # 删除之前的 train数据
    processing_num_max = 5000
    removeAllFiles("./stockdata/train")
    # 获取全部股票的日K线数据
    mkdir('./stockdata/train')
    downloader = Downloader('./stockdata/train', date_start='2010-01-01', date_end='2021-12-30', processing_num_max=processing_num_max)
    downloader.run()
    mkdir('./stockdata/test')
    # ===============================================================================
    # 测试数据的start_date需要比预想的开始测试时间早一个半月（实际往前推26个交易日就行根据数据网站接口获取交易日期数据进行计算，
    # 训练数据比较无所谓，反正数据量庞大），因为需要用更早的数据计算macd、ma、kdj等指标，计算完后再 dropna
    # 删除之前的 数据
    removeAllFiles("./stockdata/test")
    downloader = Downloader('./stockdata/test', date_start='2022-01-01', date_end='2022-03-31', processing_num_max=processing_num_max)
    downloader.run()
    sys.exit()
