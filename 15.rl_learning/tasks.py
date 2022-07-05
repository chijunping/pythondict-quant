# import os
# import pickle
# import pandas as pd
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
# from rlenv.StockTradingEnv0 import StockTradingEnv
#
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# def modelTrain(stock_file, model_path):
#     df = pd.read_csv(stock_file)
#     df = df.sort_values('date')
#     # The algorithms require a vectorized environment to run
#     env = DummyVecEnv([lambda: StockTradingEnv(df)])
#     model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log')
#     model.learn(total_timesteps=int(1e4))
#     model.save(model_path)
#
#
# def modelTest(stock_file, code):
#     model_path = "./result/model/" + code
#     day_profits = []
#     df_test = pd.read_csv(stock_file.replace('train', 'test'))
#     env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
#     model = PPO2.load(load_path=model_path, env=env)
#     obs = env.reset()
#     for i in range(len(df_test) - 1):
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         profit = env.render()
#         day_profits.append(profit)
#         if done:
#             break
#     return day_profits
#
#
# def find_file(path, name):
#     """
#     根据文件名name查找path下该文件的路径
#     :param path:
#     :param name:
#     :return:
#     """
#     # print(path, name)
#     for root, dirs, files in os.walk(path):
#         for fname in files:
#             if name in fname:
#                 return os.path.join(root, fname)
#
#
# def stock_trade(code, new_model=False):
#     stock_file = find_file('./stockdata/train', str(code))
#     model_path = "./result/model/" + code
#     day_profits = []
#     # 训练模型
#     if new_model:
#         modelTrain(stock_file, model_path)
#     elif not os.path.exists(model_path):
#         modelTrain(stock_file, model_path)
#     # 加载模型进行预测
#     df_test = pd.read_csv(stock_file.replace('train', 'test'))
#     env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
#     model = PPO2.load(load_path=model_path, env=env)
#     obs = env.reset()
#     for i in range(len(df_test) - 1):
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         profit = env.render()
#         day_profits.append(profit)
#         if done:
#             break
#     return day_profits
