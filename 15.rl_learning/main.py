import os
import pickle
import sys
import traceback

import pandas as pd
import random
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from rlenv.StockTradingEnv0 import StockTradingEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import multiprocessing

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def modelTrain(stock_file, model_path, reTrain=True):
    """
    当模型不存在，或者显示要求重建模型时，会更新模型，否则使用已存在的模型
    :param stock_file:
    :param model_path:
    :param reTrain:
    :return:
    """
    if os.path.exists(model_path) and (not reTrain):
        return
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    # model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log', seed=111, n_cpu_tf_sess=1)
    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log', seed=111)
    model.learn(total_timesteps=int(1e4))
    model.save(model_path)


def modelTest(stock_file_test, model_path):
    """
    模型预测
    :param stock_file_test:
    :param model_path:
    :return:
    """
    day_profits = []
    df_test = pd.read_csv(stock_file_test)
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    model = PPO2.load(load_path=model_path, env=env)
    obs = env.reset()
    score = 0
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        score += rewards
        if done:
            break
    return day_profits


def modelTest2(stock_file_test, model):
    """
    模型预测
    :param stock_file_test:
    :param model:
    :return:
    """
    day_profits = []
    df_test = pd.read_csv(stock_file_test)
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def find_file(path, name):
    """
    根据文件名name查找path下该文件的路径
    :param path:
    :param name:
    :return:
    """
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def stock_trade(stock_code, result_profit_path, reTrain=False):
    """
    股票交易，流程：训练或者获取模型->模型预测->返回结果
    :param code:  股票代码
    :param new_model: 是否重新训练模型
    :return:
    """
    # 训练模型
    stock_file = find_file('./stockdata/train', str(stock_code))
    stock_file_test = stock_file.replace('train', 'test')
    model_path = "./result/model/" + stock_code
    modelTrain(stock_file, model_path, reTrain)
    # 加载模型进行预测
    day_profits = modelTest(stock_file_test, model_path)
    # 追加到测试结果csv
    day_profits_df = pd.DataFrame({'stock': stock_code, 'profit': day_profits[-1]}, index=[0])
    create_or_append_to_csv(df=day_profits_df, file_path=result_profit_path, newFile=False)
    return day_profits


def stock_trade2(code):
    stock_file = find_file('./stockdata/train', str(code))
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log', gamma=0.95, n_steps=20, learning_rate=2.5e-2)
    model.learn(total_timesteps=int(1e4))

    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def test_a_stock_trade(stock_code):
    stock_file = find_file('./stockdata/train', str(stock_code))
    daily_profits = stock_trade(stock_file)
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')


def analysis_profits(resultsList):
    is_profit = [p[-1] for p in resultsList]
    len(is_profit)
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    font = fm.FontProperties(fname='./font/wqy-microhei.ttc')
    labels = 'Profit', 'Loss', '0'
    sizes = [0, 0, 0]
    for p in is_profit:
        if p > 0:
            sizes[0] += 1
        if p < 0:
            sizes[1] += 1
        else:
            sizes[2] += 1

    explode = (0.1, 0.05, 0.05)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.legend(prop=font)
    plt.show()
    plt.savefig('./img/profits.png')

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    font = fm.FontProperties(fname='./font/wqy-microhei.ttc')
    n_bins = 150

    fig, axs = plt.subplots()
    axs.hist(is_profit, bins=n_bins, density=True)
    plt.savefig('./img/profits_hist.png')


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


def removeFile(filePath):
    """
    删除文件
    :param basePath:
    :return:
    """
    if os.path.exists(filePath):
        os.remove(filePath)


def create_or_append_to_csv(df, file_path, newFile=False):
    """
    创建或追加，不存在则创建，存在则追加
    :param df:
    :param file_path:
    :return:
    """
    if os.path.exists(file_path):
        if newFile:
            os.remove(file_path)
        df.to_csv(file_path, mode='a', header=False, index=False, encoding='utf_8_sig')  # 判断一下file是否存在 > 存在：追加 / 不存在：保持
    else:
        df.to_csv(file_path, header=True, index=False, encoding='utf_8_sig')  # 判断一下file是否存在 > 存在：追加 / 不存在：保持


if __name__ == '__main__':
    result_profit_path = "result_is_profit/result_profit.csv"
    removeAllFiles("result")
    removeFile(result_profit_path)
    # 开始本来测试
    files = os.listdir("stockdata/train")
    files_test = os.listdir("stockdata/test")
    # 取交集
    all_files_list = list(set(files) & set(files_test))
    all_files_list.sort()
    # 装载结果
    day_profits_df = pd.DataFrame(columns=["stock", "profit"])
    # 进程池
    pool = multiprocessing.Pool(1)
    for file in all_files_list:
        try:
            file_df_test = pd.read_csv("stockdata/test/" + file)
            first_row_test = file_df_test.head(1)
            pctChg = first_row_test["pctChg"].values[-1]
            peTTM = first_row_test["peTTM"].values[-1]
            pbMRQ = first_row_test["pbMRQ"].values[-1]
            psTTM = first_row_test["psTTM"].values[-1]
            pcfNcfTTM = first_row_test["pcfNcfTTM"].values[-1]
            macd = first_row_test["macd"].values[-1]
            # 指标过滤，如果每个指标范围筛选都不符合，那就过滤掉
            # if (not 0.34 < pctChg < 2.16) and (not 63 < peTTM < 87) and (not 0.65 < pbMRQ < 3.85) and (not 0.03 < psTTM < 3.63) and (not -230 < pcfNcfTTM < 150):
            #     print(f"跳过股票，指标不符合：{file}")
            #     continue
            # if not ((0.3 < pctChg < 2.2) and (60 < peTTM < 90) and (0.6 < pbMRQ < 3.9) and (0 < psTTM < 3.7) and (-230 < pcfNcfTTM < 150) and (0 < macd < 0.25)):
            # if not ((-1.3 < pctChg < 2.3) and (-170 < peTTM < 90) and (-8<pbMRQ < 4) and (0 < psTTM < 3.8) and (-22 < pcfNcfTTM < 337) and (0 < macd < 0.25)):
            #     print(f"跳过股票，指标不符合：{file}")
            # continue
            print(f"符合测试要求股票：{file}")
            # 使用celery做并发
            stock_code = ".".join(file.split(".")[:3])
            # pool.apply_async(stock_trade, (stock_code, result_profit_path,)
            stock_trade(stock_code,result_profit_path,reTrain=True)
        except Exception as e:
            traceback.print_exc()
            pass
    pool.close()
    pool.join()
    # import pickle
    #
    # files = os.listdir("result")
    # print(f'实际测试股票数：{len(files)}')
    # resultsList = []
    # # 用于记录股票-收益的最终情况
    # result_profit = pd.DataFrame(columns=['stock', 'profit'])
    # for f_name in files:
    #     f = open(f"result/{f_name}", "rb")
    #     # 加载单只股票的每次交易结果，放在一个list里
    #     results = pickle.load(f)
    #     print(f'盈利股票：{f_name},盈利金额：{results[-1]}')
    #     # 将所有股票results的list对象又放在一个汇总的list里，形成resultsList
    #     resultsList.append(results)
    #     # 记录股票-最终收益情况
    #     result_profit = result_profit.append({'stock': f_name.split(".")[1], 'profit': results[-1]}, ignore_index=True)
    # print(result_profit)
    # # 删除之前的结果文件
    # if os.path.exists("result_is_profit/result_profit.csv"):
    #     os.remove("result_is_profit/result_profit.csv")
    # # 生产新的结果文件
    # result_profit.to_csv("result_is_profit/result_profit.csv", header=True, index=False, encoding='utf_8_sig')
    # # 分析收益情况
    # analysis_profits(resultsList)
    sys.exit()
