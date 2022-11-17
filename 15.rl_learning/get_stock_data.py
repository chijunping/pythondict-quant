import datetime
import sys
import traceback

import baostock as bs
import pandas as pd
import os
import multiprocessing
import talib
import numpy as np
import threading
import utils.mysql_utils as mysqlUtils
from concurrent import futures

OUTPUT = './stockdata'
host = "127.0.0.1"
port = 3306
user = "root"
password = '123456'
database = 'zeus'
columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest', 'turn',
           'sar', "ma5", "ma10", "ma20", "k", "d", "j", "talib_diff", "talib_dea",
           "talib_macd", "score"
           ]


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def stock2csv(df_stock: pd.DataFrame, path: str):
    # print("当前进程id：", os.getpid(), " 父进程id：", os.getppid())
    # print('当前线程id : %d' % threading.currentThread().ident)
    df_stock.to_csv(path, index=False)


def get_stock_data_from_db(dbInstance, code: str, start_date: str, end_date: str):
    """
    数据源读取，供框架计算使用
    :param dbInstance:
    :param code:
    :param start_date:
    :param end_date:
    :return:
    """
    # 读取本地csv数据
    # data_path = os.path.abspath(os.path.dirname(__file__) + "../../../data/csv/" + code + ".csv")
    # print(f"csv路径：{data_path}")
    # data = pd.read_csv(data_path, index_col='date')
    sqlStr = f"select * from quant_stock_data where `code`='{code}' and `date`>='{start_date}' and `date`<='{end_date}' order by `date`"
    data = mysqlUtils.read_data(dbInstance=dbInstance, sql=sqlStr)
    data.index = [datetime.datetime.strptime(i, "%Y%m%d") for i in data['date']]
    # 国内市场不需要此字段，所以赋值为0即可
    data['openinterest'] = 0
    data = data[columns]
    # data.dropna(inplace=True)
    # data["ps_ttm"] = data["ps_ttm"].fillna(0)
    return data


def get_stock_data_to_csv(stock_code, stock_name, start_date, end_date, output_dir):
    try:
        # adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权
        conn = mysqlUtils.connect_database(host, port, user, password, database)
        df_stock = get_stock_data_from_db(dbInstance=conn, code=stock_code, start_date=start_date, end_date=end_date)
        conn.close()
        path = f'{output_dir}/{stock_code}.{stock_name.strip("*")}.csv'
        stock2csv(df_stock, path)
        # 调用baostock获取股票数据，后续可以改成akshare获取数据
        print(f'数据获取成功 {stock_code} {stock_name}')
    except Exception as e:
        traceback.print_exc()
        raise


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

    def get_stock_info_by_date(self, date):
        conn = mysqlUtils.connect_database(host, port, user, password, database)
        codeSql = f"""
        select DISTINCT 
            t1.`code`,t1.`name` 
        FROM
            quant_stock_code_craw t1
        join(select DISTINCT `code` from quant_stock_data where `date`='20221104') t2 on t1.`code`=t2.`code` order by `code`
        """
        stockInfo = mysqlUtils.read_data(conn, codeSql)
        conn.close()
        return stockInfo

    def run(self):
        stock_info = self.get_stock_info_by_date(self.date_end)
        # 股票计数器
        processing_num = 0
        pool = futures.ProcessPoolExecutor(max_workers=6)
        all_task = []
        for index, row in stock_info.iterrows():
            # 过滤股票名字带*号的，影响windows创建文件
            stock_name = row["name"]
            stock_code = row["code"]
            if '*' in stock_name:
                continue
            if 'ST' in stock_name:
                continue
            # 过滤可能退市的股票
            if '退' in stock_name:
                continue
            # 控制股票数：方便测试时候控制股票数量，避免拉取太多数据
            processing_num = processing_num + 1
            if processing_num >= self.processing_num_max:
                break
            # 获取数据
            task = pool.submit(get_stock_data_to_csv, stock_code, stock_name, self.date_start, self.date_end, self.output_dir)
            all_task.append(task)
        futures.wait(fs=all_task, return_when=futures.ALL_COMPLETED)
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
    processing_num_max = 100
    removeAllFiles("./stockdata/train")
    # 获取全部股票的日K线数据
    mkdir('./stockdata/train')
    downloader = Downloader('./stockdata/train', date_start='2000-01-01', date_end='2021-12-30', processing_num_max=processing_num_max)
    downloader.run()
    # ===============================================================================
    # 测试数据的start_date需要比预想的开始测试时间早一个半月（实际往前推26个交易日就行根据数据网站接口获取交易日期数据进行计算，
    # 训练数据比较无所谓，反正数据量庞大），因为需要用更早的数据计算macd、ma、kdj等指标，计算完后再 dropna
    # 删除之前的 数据
    removeAllFiles("./stockdata/test")
    mkdir('./stockdata/test')
    downloader = Downloader('./stockdata/test', date_start='2021-12-01', date_end='2022-08-05', processing_num_max=processing_num_max)
    downloader.run()
    sys.exit()
