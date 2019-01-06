# -*- coding: UTF-8 -*-
"""
Author: Xiandong QI
Date: 2018-11-11
Email: xqiad@connect.ust.hk
Description: A StockData class as the code test of 衍盛中国
"""
# Built-in/Generic Imports
import os
from collections import OrderedDict

# Libs
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Own modules


class StockData(object):
    """This is StockData."""

    def __init__(self, path=None):
        """初始化函数，path为csv数据文件所在路径
        """
        self.path = path or './data/'
        self.stock_data = {}  # {symbol_1: symbol_1_df, symbol_2: symbol_2_df, ...}
        self.all_symbols = {}
        # get all candidate symbols and its region info
        # {'300399': 'SZ', '600130': 'SH', '600626': 'SH', ...}
        for file_name in os.listdir(self.path):
            symbol = file_name.split('.')[0]
            symbol_region = file_name.split('.')[1]
            self.all_symbols[symbol] = symbol_region

    def read(self, symbols):
        """symbols是股票代码列表，包括1~n个股票代码，函数从对应symbol的文件中读取相
        数据并选取合适结构保存在`类`中
        例如： stock_data.read(['600626', '600999'])
        """
        for symbol in symbols:
            symbol_file_name = str(
                symbol + "." + self.all_symbols[symbol] + ".csv")
            # print("symbol_file_name: ", symbol_file_name)
            # Output:
            # symbol_file_name:  ['600626.SH.csv']
            symbol_path = str(self.path + symbol_file_name)
            # print("symbol_path: ", symbol_path)
            df = pd.read_csv(symbol_path)
            # the first column is `date`, we need to add it
            df.rename(columns={"TRADE_DT": "date"}, inplace=True)
            # use `pd.to_numeric`, so that we can compare the values of df
            self.stock_data[symbol] = df.apply(pd.to_numeric, errors='ignore')

    def get_data_by_symbol(self, symbol, start_date, end_date):
        """获取某个symbol从start_date 到end_date之间的所有日频数据
        返回pandas.DataFrame结构
        """
        df = self.stock_data[symbol]
        df = df[(df['date'] >= int(start_date))
                & (df['date'] <= int(end_date))]
        df.set_index('date', inplace=True)
        return df

    def get_data_by_date(self, adate, symbols):
        """获取某一天中，对应symbols的所有日频数据
        返回pandas.DataFrame结构
        """
        df_list = []
        for symbol in symbols:
            df = self.stock_data[symbol]
            df = df[(df['date'] == int(adate))]
            # print(df['date'])
            df_list.append(df)
        df = pd.concat(df_list)
        df.set_index('S_INFO_WINDCODE', inplace=True)
        return df

    def get_data_by_field(self, field, symbols):
        """获取symbols在某个field上所有的所有交易日的数据
        返回pandas.DataFrame结构
        """
        df_list = []
        for symbol in symbols:
            df = self.stock_data[symbol]
            df = df[[field, 'date']]
            df.set_index('date', inplace=True)
            df.rename(columns={field: symbol}, inplace=True)
            # combine with 'date' as index
            # print(df)
            df_list.append(df)
        df = pd.concat(df_list, sort=True)
        return df

    ###########
    # 采样和转换
    ###########

    def format_date(self, symbol):
        """symbol的数据中date在读入时，可能是int或者string类型，请将这一列中的每个date转化为datetime类型（或者pandas.Timestamp类型）
        """
        df = self.stock_data[symbol]
        df['date'] = df['date'].apply(
            lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        self.stock_data[symbol] = df

    def plot(self, symbol, field=['S_DQ_CLOSE', 'S_DQ_VOLUME']):
        """做出symbol关于field的走势图，field取open，high，low，close, vwap中的某一个,走势图为折线图，
        取volume或者turnover中的某一个时，走势图为柱状图。
        简单画一下就行，类似于一般股票网站上价量的图。
        """
        df = self.stock_data[symbol]
        for a_field in field:
            if a_field in ['S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_VWAP']:
                plt.figure()
                df_0 = df[a_field]
                df_0.plot()
                plt.savefig('figure_{}.png'.format(a_field))
            elif a_field in ['S_DQ_VOLUME', 'S_DQ_AMOUNT']:
                plt.figure()
                df_1 = df[a_field]
                df_1.plot.bar()
                plt.savefig('figure_{}.png'.format(a_field))
            else:
                raise Exception('Nonexistent Field')

    def adjust_data(self, symbol):
        """对代表价格（open, high, low, close）进行（从后向）前复权。计算方法见doc文档。
        Note: 对于每一天的价格均要复权。
        Note: 也可用 pd.Series.shift 的方法，直接进行series和series之间的运算
        Reference: https://joshschertz.com/2016/08/27/Vectorizing-Adjusted-Close-with-Python/
        """
        df = self.stock_data[symbol]
        columns = ['S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']
        for column in columns:
            adj_column = 'S_DQ_ADJ' + column.split('_')[-1]
            # 'S_DQ_HIGH' -> 'S_DQ_ADJHIGH'

            price_col = df[column].values  # 原价格
            preprice_col = df['S_DQ_PRECLOSE'].values  # 昨收盘价
            adj_price_col = np.zeros(len(df.index))  # 复权后价格
            adj_price_col[0] = price_col[0]
            adj_factor = np.ones(len(df.index))  # 复权因子

            for i in range(1, len(price_col)):
                adj_factor[i] = round(
                    price_col[i - 1] / preprice_col[i] * adj_factor[i - 1], 2)
                adj_price_col[i] = round(price_col[i] * adj_factor[i], 2)

            df[adj_column] = adj_price_col  # 复权价（后复权）
            df[adj_column] = df[adj_column] / adj_factor[-1]  # 复权价（前复权）
            # print(adj_factor[-1])
            # 1.38 1.5924916914057122e+53 0.0 7.55
            # print(df[adj_column])

    def resample(self, symbol, freq=5):
        """将symbol的日频数据进行sample（采样），freq表示天数。
        采样只需完成open，high，low，close，volume，turnover，vwap这些field即可

        例如：StockData  data
        df =data.resample(“600000”, 5) 表示对600000的日频数据进行5天的采样，其中
        open，close表示这5天第一日开盘价和最后一日的收盘价
        high，low表示这5天的最高价和最低价
        volume，turnover 表示这5天的总成交量，总成交额
        vwap表示这5天的成交量加权平均价格

        返回dataframe
        """
        freq = str(freq) + 'D'
        df = self.stock_data[symbol]
        df = df[["S_DQ_OPEN", 'S_DQ_HIGH', 'S_DQ_LOW',
                 'S_DQ_CLOSE', "S_DQ_VOLUME", "S_DQ_AMOUNT", "S_DQ_AVGPRICE"]]
        resample_df = df.resample(freq).agg(
            OrderedDict([
                ('S_DQ_OPEN', 'first'),
                ('S_DQ_HIGH', 'max'),
                ('S_DQ_LOW', 'min'),
                ('S_DQ_CLOSE', 'last'),
                ('S_DQ_VOLUME', 'sum'),
                ('S_DQ_AMOUNT', 'sum'),
            ]))
        resample_df["S_DQ_AVGPRICE"] = df['S_DQ_AMOUNT'] / df['S_DQ_VOLUME']
        return resample_df

    #####################
    # Computational Tools
    #####################

    # Notice:
    # 1. 所用到的数据，是在E2.3中前复权过的数据，缺失或者停牌数据不处理。
    # 2. 很多指标由于涉及滑动窗口，需要做一些rolling的计算，请参考pandas相关文档。

    def moving_average(self, symbol, field, window=5):
        """计算symbol对应field的移动平均， window表示滑动窗口。
        请分别测试当窗口window为5，20，60的时候的计算结果
        返回一个pandas.Series对象，index是交易日（date）

        可以用self.plot函数画图看下结果。
        """
        df = self.stock_data[symbol]
        df = df[[field]]
        # Notice: should be changed to "S_DQ_ADJOPEN"
        short_rolling = df.rolling(window=window).mean()
        # long_rolling = df.rolling(window=window * 10).mean()
        # 均返回一个pandas.Series对象
        return short_rolling

    def ema(self, symbol, field, params=None):
        """计算symbol日频数据的ema指标
        返回一个pandas.Series对象，index是交易日（date）

        # Using Pandas to calculate a 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.
        """
        df = self.stock_data[symbol]
        df = df[[field]]
        # Notice: should be changed to "S_DQ_ADJOPEN"
        ema_short = df.ewm(span=20, adjust=False).mean()
        return ema_short

    def atr(self, symbol, params=None):
        """计算symbol日频数据的atr指标
        返回一个pandas.Series对象，index是交易日（date）

        Reference: http://kaushik316-blog.logdown.com/posts/1964522
        """
        df = self.stock_data[symbol]
        df = df[['S_DQ_ADJHIGH', 'S_DQ_ADJLOW',
                 'S_DQ_ADJCLOSE']]
        # Notice: should be changed to "S_DQ_ADJOPEN" and so on...
        atr = talib.ATR(df['S_DQ_ADJHIGH'], df['S_DQ_ADJLOW'],
                        df['S_DQ_ADJCLOSE'], timeperiod=14)
        return atr

    def rsi(self, symbol, params):
        """计算symbol日频数据的rsi指标
        返回一个pandas.Series对象，index是交易日（date）
        """
        df = self.stock_data[symbol]
        df = df[['S_DQ_ADJCLOSE']]
        rsi = talib.RSI(df['S_DQ_ADJCLOSE'], timeperiod=14)
        return rsi

    def macd(self, symbol, params=None):
        """计算symbol日频数据的macd指标
        返回一个pandas.Series对象，index是交易日（date）
        """
        df = self.stock_data[symbol]
        df = df[['S_DQ_ADJCLOSE']]
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        df['macd'], df['macdSignal'], df['macdHist'] = talib.MACD(
            df['S_DQ_ADJCLOSE'], fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
        return df


if __name__ == "__main__":
    stock_data = StockData()
    # print("stock_data.all_symbols: ", stock_data.all_symbols)

    stock_data.read(symbols=['600626', '600999'])
    # print(stock_data.stock_data['600626'])
    # print(stock_data.stock_data['600626'].columns)

    # data_by_symbol = stock_data.get_data_by_symbol(
    #     symbol='600626', start_date='20121030', end_date='20141216')
    # print(data_by_symbol)

    # data_by_date = stock_data.get_data_by_date(
    #     adate='20121030', symbols=['600626', '600999'])
    # print(data_by_date)

    # data_by_field = stock_data.get_data_by_field(
    #     field="S_DQ_CLOSE", symbols=['600626', '600999'])
    # print(data_by_field)

    ###########
    # 采样和转换
    ###########

    stock_data.format_date(symbol='600626')
    # print(stock_data.stock_data['600626'])

    # # Notice: 此刻，我们默认 df.index 为 date
    # save_plot = stock_data.plot(symbol='600626', field=[
    #                             'S_DQ_CLOSE', 'S_DQ_VOLUME'])

    # # 计算复权价格，并补充在df内
    # stock_data.adjust_data(symbol='600626')
    # df = stock_data.stock_data['600626']
    # print(df['S_DQ_ADJCLOSE'])

    # # Notice: 此刻，我们默认 df.index 为 date
    # resample_df = stock_data.resample(symbol='600626', freq=5)
    # print(resample_df)

    #####################
    # Computational Tools
    #####################

    # moving_average = stock_data.moving_average(
    #     symbol='600626', field='S_DQ_ADJOPEN', window=5)
    # print(moving_average)

    # ema = stock_data.ema(symbol='600626', field='S_DQ_ADJOPEN', params=None)
    # print(ema)

    # atr = stock_data.atr(symbol='600626', params=None)
    # print(atr)

    # rsi = stock_data.rsi(symbol='600626', params=None)
    # print(rsi)

    # macd = stock_data.macd(symbol='600626', params=None)
    # print(macd)
