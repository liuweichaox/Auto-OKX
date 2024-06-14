import json
import time
import logging
import numpy as np
import redis
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.PublicData as PublicData
import okx.Funding as Funding
from ratelimit import limits, sleep_and_retry
from datetime import datetime
from config import Config
from fake_config import FakeConfig


class TradingStrategy:
    def __init__(self, flag="1"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",
            filemode="w",
        )
        self.logger = logging.getLogger()
        if flag == "0":
            self.config = Config()
        else:
            self.config = FakeConfig()

        self.redis_client = redis.StrictRedis(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            db=self.config.REDIS_DB,
            password=self.config.REDIS_PASSWORD,
        )
        api_config = {
            "api_key": self.config.API_KEY,
            "api_secret_key": self.config.SECRET_KEY,
            "passphrase": self.config.PASSPHRASE,
            "use_server_time": False,
            "domain": self.config.BASE_URL,
            "debug": self.config.DEBUG,
            "proxy": None,
        }
        self.accountAPI = Account.AccountAPI(**api_config)
        self.marketAPI = MarketData.MarketAPI(**api_config)
        self.tradeAPI = Trade.TradeAPI(**api_config)
        self.publicAPI = PublicData.PublicAPI(**api_config)
        self.fundingAPI = Funding.FundingAPI(**api_config)

    @sleep_and_retry
    @limits(calls=20, period=2)
    def get_price_limit(self, symbol):
        # 获取限价 20次/2s
        response = self.publicAPI.get_price_limit(symbol)
        buyLmt = response["data"][0]["buyLmt"]
        buyLmt = float(buyLmt)
        return buyLmt

    @sleep_and_retry
    @limits(calls=1, period=2)
    def get_exchange_rate(self):
        # 获取汇率信息
        response = self.marketAPI.get_exchange_rate()
        usdCny = response["data"][0]["usdCny"]
        return float(usdCny)

    @sleep_and_retry
    @limits(calls=20, period=2)
    def get_current_price(self, symbol):
        # 获取当前价格
        response = self.marketAPI.get_ticker(symbol)
        last = response["data"][0]["last"]
        return float(last)

    @sleep_and_retry
    @limits(calls=40, period=2)
    def get_candlesticks(self, symbol, bar, after=None):
        # 获取K线数据。K线数据按请求的粒度分组返回，K线数据每个粒度最多可获取最近1,440条。 40次/2s
        return self.marketAPI.get_candlesticks(
            instId=symbol,
            after=after,
            bar=bar,
            limit="300",
        )

    def get_latest_candlestick_data(self, instId, bar):
        total_data = []
        after = ""
        while len(total_data) < 1440:
            data = self.get_candlesticks(instId, bar, after)
            data = data["data"]
            if data is None or len(data) == 0:
                break
            total_data.extend(data)
            after = data[-1][0]
        return total_data

    @sleep_and_retry
    @limits(calls=20, period=2)
    def get_history_candlesticks(self, symbol, end_timestamp, bar):
        # 获取最近几年的历史k线数据(1s k线支持查询最近3个月的数据) 20次/2s
        # 时间是从后往前找，所以是以最大的时间为准，向前查找
        return self.marketAPI.get_history_candlesticks(
            instId=symbol,
            after=str(end_timestamp),
            bar=bar,
            limit="100",
        )

    def get_price_data(self, symbol, timeframe):
        """
        获取指定交易对和时间范围的价格数据。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。
        timeframe (str): 时间范围，例如"1D"表示每日。

        返回:
        DataFrame: 包含价格数据的DataFrame。
        """
        data = self.get_latest_candlestick_data(symbol, timeframe)

        columns = [
            "timestamp",  # 开始时间
            "open",  # 开盘价格
            "high",  # 最高价格
            "low",  # 最低价格
            "close",  # 收盘价格
            "volume",  # 交易量，以张为单位
        ]
        data_sliced = [row[:6] for row in data]
        df = pd.DataFrame(data_sliced, columns=columns)

        df["timestamp"] = (
            pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")
            .dt.tz_localize("UTC")
            .dt.tz_convert("Asia/Shanghai")
        )

        df.set_index("timestamp", inplace=True)

        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].apply(pd.to_numeric)
        df.sort_index(inplace=True)
        return df

    def feature_engineering(self, df):
        """
        对价格数据进行特征工程处理。

        参数:
        df (DataFrame): 包含价格数据的DataFrame。

        返回:
        DataFrame: 处理后的DataFrame。
        """
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        indicator_bb = ta.volatility.BollingerBands(
            close=df["close"], window=20, window_dev=2
        )
        df["bb_bbm"] = indicator_bb.bollinger_mavg()
        df["bb_bbh"] = indicator_bb.bollinger_hband()
        df["bb_bbl"] = indicator_bb.bollinger_lband()
        df["returns"] = df["close"].pct_change()

        df = self.calculate_macd(df)
        df = self.calculate_atr(df)
        df = self.calculate_adx(df)
        df.dropna(inplace=True)
        return df

    def fetch_feature(self, price_data):
        X = price_data[
            [
                "open",
                "high",
                "low",
                "volume",
                "rsi",
                "bb_bbm",
                "bb_bbh",
                "bb_bbl",
                "returns",
                "macd",
                "macd_signal",
                "atr",
                "adx",
            ]
        ]
        y = price_data["close"]
        return X, y

    def calculate_atr(self, df, window=14):
        """
        计算平均真实范围（ATR）指标。

        参数:
        df (DataFrame): 包含价格数据的DataFrame。
        window (int): ATR计算窗口大小，默认为14。

        返回:
        DataFrame: 包含ATR指标的DataFrame。
        """
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=window
        )
        return df

    def calculate_macd(self, df):
        """
        计算MACD指标。

        参数:
        df (DataFrame): 包含价格数据的DataFrame。

        返回:
        DataFrame: 包含MACD指标的DataFrame。
        """
        df["macd"] = ta.trend.macd(df["close"])
        df["macd_signal"] = ta.trend.macd_signal(df["close"])
        df["macd_diff"] = df["macd"] - df["macd_signal"]
        return df

    def calculate_adx(self, df, window=14):
        """
        计算平均趋向指数（ADX）指标。

        参数:
        df (DataFrame): 包含价格数据的DataFrame。
        window (int): ADX计算窗口大小，默认为14。

        返回:
        DataFrame: 包含ADX指标的DataFrame。
        """
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=window)
        return df

    def sentiment_analysis(self, text):
        """
        对文本进行情感分析。

        参数:
        text (str): 要分析的文本。

        返回:
        float: 情感分数。
        """
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        return sentiment["compound"]

    def calculate_optimal_trade_size(
        self, account_balance, risk_percentage, atr, stop_loss_pips
    ):
        """
        计算最优交易量。

        参数:
        account_balance (float): 账户余额。
        risk_percentage (float): 风险百分比。
        atr (float): ATR指标值。
        stop_loss_pips (float): 止损点数。

        返回:
        float: 最优交易量。
        """
        risk_amount = account_balance * risk_percentage
        trade_size = risk_amount / (atr / stop_loss_pips)
        return round(trade_size, 4)

    def train_models(self, price_data):
        """
        训练线性回归和随机森林模型。

        参数:
        X (DataFrame): 特征数据。
        y (Series): 目标数据。

        返回:
        tuple: 线性回归和随机森林模型。
        """
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor()

        X, y = strategy.fetch_feature(price_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        lr_score = lr_model.score(X_test, y_test)
        rf_score = rf_model.score(X_test, y_test)

        print(f"线性回归模型得分: {lr_score}")
        print(f"随机森林模型得分: {rf_score}")

        return lr_model, rf_model

    def execute_combined_trading_strategy(
        self,
        symbol,
        account_balance,
        risk_percentage,
        max_loss_limit,
    ):
        """
        执行组合交易策略。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。
        account_balance (float): 账户余额。
        risk_percentage (float): 风险百分比。
        max_loss_limit (float): 最大损失比例。

        返回:
        None
        """
        price_data = self.get_price_data(symbol, "1m")
        price_data = self.feature_engineering(price_data)
        # 训练模型
        lr_model, rf_model = strategy.train_models(price_data)

        latest_data = price_data.iloc[-1]
        latest_features = latest_data[
            [
                "open",
                "high",
                "low",
                "volume",
                "rsi",
                "bb_bbm",
                "bb_bbh",
                "bb_bbl",
                "returns",
                "macd",
                "macd_signal",
                "atr",
                "adx",
            ]
        ].values.reshape(1, -1)

        latest_features_df = pd.DataFrame(
            latest_features,
            columns=[
                "open",
                "high",
                "low",
                "volume",
                "rsi",
                "bb_bbm",
                "bb_bbh",
                "bb_bbl",
                "returns",
                "macd",
                "macd_signal",
                "atr",
                "adx",
            ],
        )
        lr_pred = lr_model.predict(latest_features_df)[0]
        rf_pred = rf_model.predict(latest_features_df)[0]

        sentiment_text = (
            "Bitcoin price is soaring due to increased institutional interest."
        )
        sentiment_score = self.sentiment_analysis(sentiment_text)

        avg_pred = self.combined_prediction(lr_pred, rf_pred, sentiment_score)

        is_buy_condition = self.check_buy_condition(price_data)
        is_sell_condition = self.check_sell_condition(price_data)

        entry_price = latest_data["close"]
        atr = latest_data["atr"]
        rsi = latest_data["rsi"]
        macd_diff = latest_data["macd_diff"]
        adx = latest_data["adx"]

        take_profit_price, stop_loss_price = self.dynamic_take_profit_and_stop_loss(
            entry_price, atr, macd_diff, rsi, adx
        )
        optimal_trade_size = self.calculate_optimal_trade_size(
            account_balance, risk_percentage, atr, entry_price
        )

        # 打印详细的检查信息
        print(f"情绪分数: {sentiment_score}")
        print(f"平均预测: {avg_pred}")
        print(f"买入条件: {is_buy_condition}")
        print(f"卖出条件: {is_sell_condition}")
        print(f"入场价格 (买入/卖出价格): {entry_price}")
        print(f"RSI (相对强弱指数): {rsi}")
        print(f"止盈价格: {take_profit_price}")
        print(f"止损价格: {stop_loss_price}")
        print(f"最优交易量: {optimal_trade_size}")

        if is_buy_condition and avg_pred > entry_price:
            print("满足买入条件...")
            if not self.check_order_quantity(symbol, "buy", optimal_trade_size):
                optimal_trade_size = self.get_max_buy_size(
                    symbol,
                )
            response = self.place_order(symbol, "buy", optimal_trade_size)
            print(response)
            self.monitor_position(
                symbol,
                entry_price,
                take_profit_price,
                stop_loss_price,
                "buy",
                optimal_trade_size,
                max_loss_limit,
            )
        elif is_sell_condition and avg_pred < entry_price:
            print("满足卖出条件...")
            if not self.check_order_quantity(symbol, "sell", optimal_trade_size):
                optimal_trade_size = self.get_max_sell_size(
                    symbol,
                )
            response = self.place_order(symbol, "sell", optimal_trade_size)
            print(response)
            self.monitor_position(
                symbol,
                entry_price,
                take_profit_price,
                stop_loss_price,
                "sell",
                optimal_trade_size,
                max_loss_limit,
            )

        else:
            print("买入条件和卖出条件均不满足，无交易动作。")

    def combined_prediction(self, lr_pred, rf_pred, sentiment_score):
        """
        计算组合预测值。

        参数:
        lr_pred (float): 线性回归预测值。
        rf_pred (float): 随机森林预测值。
        sentiment_score (float): 情感分数。

        返回:
        float: 组合预测值。
        """
        return (lr_pred + rf_pred + sentiment_score) / 3

    def check_buy_condition(self, df):
        """
        检查是否满足买入条件。

        参数:
        df (DataFrame): 包含价格数据的DataFrame。

        返回:
        bool: 是否满足买入条件。
        """
        return df["rsi"].iloc[-1] < 40

    def check_sell_condition(self, df):
        """
        检查是否满足卖出条件。

        参数:
        df (DataFrame): 包含价格数据的DataFrame。

        返回:
        bool: 是否满足卖出条件。
        """
        return df["rsi"].iloc[-1] > 60

    def dynamic_take_profit_and_stop_loss(
        self,
        entry_price,
        atr,
        trend_strength,
        rsi,
        adx,
        base_take_profit_ratio=0.05,
        base_stop_loss_ratio=0.02,
    ):
        """
        计算动态止盈止损价格。

        参数:
        entry_price (float): 买入/卖出价格。
        atr (float): ATR指标值。
        trend_strength (float): 趋势强度。
        rsi (float): RSI指标值。
        adx (float): ADX指标值。
        base_take_profit_ratio (float): 基础止盈比例，默认为0.05。
        base_stop_loss_ratio (float): 基础止损比例，默认为0.02。

        返回:
        tuple: 止盈价格和止损价格。
        """
        # 基于RSI调整止盈止损价格
        if rsi < 40:
            take_profit_price = entry_price * (1 + base_take_profit_ratio + atr)
            stop_loss_price = entry_price * (1 - base_stop_loss_ratio - atr)
        elif rsi > 60:
            take_profit_price = entry_price * (1 + base_take_profit_ratio - atr)
            stop_loss_price = entry_price * (1 - base_stop_loss_ratio + atr)
        else:
            take_profit_price = entry_price * (1 + base_take_profit_ratio)
            stop_loss_price = entry_price * (1 - base_stop_loss_ratio)

        # 基于ADX调整止盈止损价格
        if adx > 25:
            take_profit_price *= 1.1
            stop_loss_price *= 0.9

        # 基于趋势强度调整止盈止损价格
        if trend_strength > 0:
            take_profit_price *= 1 + trend_strength
            stop_loss_price *= 1 - trend_strength
        elif trend_strength < 0:
            take_profit_price *= 1 - abs(trend_strength)
            stop_loss_price *= 1 + abs(trend_strength)

        return round(take_profit_price, 4), round(stop_loss_price, 4)

    def monitor_position(
        self,
        symbol,
        entry_price,
        take_profit_price,
        stop_loss_price,
        position_type,
        quantity,
        max_loss_limit,
    ):
        """
        监控持仓并执行止盈止损操作。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。
        entry_price (float): 买入/卖出价格。
        take_profit_price (float): 止盈价格。
        stop_loss_price (float): 止损价格。
        position_type (str): 持仓类型，"buy"或"sell"。
        quantity (float): 交易量。
        max_loss_limit (float): 最大损失比例。

        返回:
        None
        """
        while True:
            print("进入监控持仓并执行止盈止损操作")
            price_data = self.get_price_data(symbol, "1m")
            latest_data = price_data.iloc[-1]
            current_price = latest_data["close"]

            print(f"止盈价格: {take_profit_price}")
            print(f"止损价格: {stop_loss_price}")
            print(f"(买入/卖出价格): {entry_price}")

            if position_type == "buy":
                if not self.check_order_quantity(symbol, "sell", quantity):
                    quantity = self.get_max_sell_size(
                        symbol,
                    )
                if current_price >= take_profit_price:
                    print("达到止盈价格，卖出...")
                    response = self.place_order(symbol, "sell", quantity)
                    print(response)
                    break
                elif current_price <= stop_loss_price:
                    print("达到止损价格，卖出...")
                    response = self.place_order(symbol, "sell", quantity)
                    print(response)
                    break
                elif (entry_price - current_price) / entry_price >= max_loss_limit:
                    print("达到最大损失限制，卖出...")
                    response = self.place_order(symbol, "sell", quantity)
                    print(response)
                    break
            elif position_type == "sell":
                if not self.check_order_quantity(symbol, "buy", quantity):
                    quantity = self.get_max_sell_size(
                        symbol,
                    )
                if current_price <= take_profit_price:
                    print("达到止盈价格，买入...")
                    response = self.place_order(symbol, "buy", quantity)
                    print(response)
                    break
                elif current_price >= stop_loss_price:
                    print("达到止损价格，买入...")
                    response = self.place_order(symbol, "buy", quantity)
                    print(response)
                    break
                elif (current_price - entry_price) / entry_price >= max_loss_limit:
                    print("达到最大损失限制，买入...")
                    response = self.place_order(symbol, "buy", quantity)
                    print(response)
                    break
            print("继续下一轮监控持仓并执行止盈止损操作")
            time.sleep(5)

    def check_order_quantity(self, symbol, order_type, quantity):
        """
        检查订单数量是否符合交易所要求。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。
        order_type (str): 订单类型，"buy"或"sell"。
        quantity (float): 订单数量。
        price (float): 订单价格。

        返回:
        bool: 订单数量是否符合要求。
        """
        # 获取最小交易量、最大买入量和最大卖出量，这里只是示例
        min_trade_size = self.get_min_trade_size(symbol)
        max_buy_size = self.get_max_buy_size(symbol)
        max_sell_size = self.get_max_sell_size(symbol)

        if order_type == "buy":
            return min_trade_size <= quantity <= max_buy_size
        elif order_type == "sell":
            return min_trade_size <= quantity <= max_sell_size
        else:
            return False

    @sleep_and_retry
    @limits(calls=20, period=2)
    def get_instruments(self, symbol):
        # 获取交易产品基础信息 20次/2s
        return self.publicAPI.get_instruments(instType="SPOT", instId=symbol)

    # 真实环境中需要实现获取最小交易量、最大买入量和最大卖出量的方法
    def get_min_trade_size(self, symbol):
        """
        获取最小交易量。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。

        返回:
        float: 最小交易量。
        """
        # 实现获取最小交易量的逻辑，这里只是示例
        response = self.get_instruments(symbol)
        minSz = response["data"][0]["minSz"]
        minSz = float(minSz)
        return minSz

    def get_max_buy_size(self, symbol):
        """
        获取最大买入量。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。

        返回:
        float: 最大买入量。
        """
        # 实现获取最大买入量的逻辑，这里只是示例
        response = self.get_max_avail_size(symbol)
        availBuy = response["data"][0]["availBuy"]
        availBuy = float(availBuy)
        return round(availBuy, 4)

    @sleep_and_retry
    @limits(calls=20, period=2)
    def get_max_avail_size(self, symbol):
        # 获取最大可用数量 20次/2s
        return self.accountAPI.get_max_avail_size(instId=symbol, tdMode="cash")

    def get_max_sell_size(self, symbol):
        """
        获取最大卖出量。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。
        price (float): 当前价格。

        返回:
        float: 最大卖出量。
        """
        # 实现获取最大卖出量的逻辑，这里只是示例
        response = self.get_max_avail_size(symbol)
        availSell = response["data"][0]["availSell"]
        availSell = float(availSell)
        return round(availSell, 4)

    # 以下为交易所相关操作的示例，需根据实际情况实现
    @sleep_and_retry
    @limits(calls=60, period=2)
    def place_order(self, symbol, order_type, quantity):
        """
        下单操作。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。
        order_type (str): 订单类型，"buy"或"sell"。
        quantity (float): 交易量。

        返回:
        str: 下单结果。
        """
        # 实现下单逻辑

        return self.tradeAPI.place_order(
            instId=symbol,
            tdMode="cash",
            side=order_type,
            ordType="market",
            sz=str(quantity),
        )

    # 真实环境中需要实现获取账户余额的方法
    def get_account_balance(self, ccy=""):
        """
        获取账户余额。

        返回:
        float: 账户余额。
        """
        return self.accountAPI.get_account_balance(ccy)

    def run_trading_bot(self, symbol, account_balance, risk_percentage, max_loss_limit):
        """
        运行交易机器人，持续执行交易策略。

        参数:
        symbol (str): 交易对符号，例如"BTC-USDT"。
        account_balance (float): 账户余额。
        risk_percentage (float): 风险百分比。
        max_loss_limit (float): 最大损失比例。

        返回:
        None
        """
        while True:
            try:
                self.execute_combined_trading_strategy(
                    symbol,
                    account_balance,
                    risk_percentage,
                    max_loss_limit,
                )
            except Exception as e:
                self.logger.error(e)
                print(f"发生错误: {e}")

            # 等待再次执行策略
            time.sleep(5)


print("程序开始执行...")
account_balance = 1000  # 账户余额
risk_percentage = 0.01  # 风险比例
max_loss_limit = 0.02  # 最大损失比例为2%
symbol = "DOGE-USDT"
strategy = TradingStrategy("1")
strategy.run_trading_bot(symbol, account_balance, risk_percentage, max_loss_limit)
