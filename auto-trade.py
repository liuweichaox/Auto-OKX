import json
import time
import logging
import numpy as np
import redis
from sklearn.linear_model import LinearRegression
import datetime
import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.PublicData as PublicData
import okx.Funding as Funding
from ratelimit import limits, sleep_and_retry
import traceback

# 初始化日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="w",
)
logger = logging.getLogger("auto-trades")


# 实盘交易设置
FLAG = "0"  # 实盘: 0 , 模拟盘：1
# API Key等信息
if FLAG == "0":
    BASE_URL = "https://www.okx.com"
    API_KEY = "f52b2961-8c08-4af4-876c-d4c6bcebdc6c"
    SECRET_KEY = "7DB206F3D875F9062170D14B1BC23BEF"
    PASSPHRASE = "Lwc1st+-"
    redis_client = redis.StrictRedis(
        host="localhost", port=6379, db=0, password="123456"
    )
elif FLAG == "1":
    BASE_URL = "https://www.okx.com"
    API_KEY = "12648afa-8e43-4d58-87f3-1a1510698ce2"
    SECRET_KEY = "D9A798DF9EBC04954835D887B577386F"
    PASSPHRASE = "Lwc1st+-"
    redis_client = redis.StrictRedis(
        host="localhost", port=6379, db=1, password="123456"
    )


accountAPI = Account.AccountAPI(
    API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=False
)
marketAPI = MarketData.MarketAPI(
    API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=False
)
tradeAPI = Trade.TradeAPI(
    API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=False
)
publicAPI = PublicData.PublicAPI(
    API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=False
)
fundingAPI = Funding.FundingAPI(
    API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=False
)


@sleep_and_retry
@limits(calls=20, period=2)
def get_product_info(ccy):
    # 获取交易产品基础信息 20次/2s
    response = publicAPI.get_instruments(instType="SPOT", instId=ccy)
    return response["data"][0]


@sleep_and_retry
@limits(calls=1, period=2)
def get_exchange_rate():
    # 获取汇率信息
    response = marketAPI.get_exchange_rate()
    usdCny = response["data"][0]["usdCny"]
    return float(usdCny)


@sleep_and_retry
@limits(calls=20, period=2)
def get_current_price(ccy):
    # 获取当前价格
    response = marketAPI.get_ticker(ccy)
    last = response["data"][0]["last"]
    return float(last)


def calculate_kelly(risk_reward_ratio, win_probability):
    return (
        risk_reward_ratio * win_probability - (1 - win_probability)
    ) / risk_reward_ratio


def calculate_trade_size(
    product_info,
    exchange_rate,
    current_price,
    stop_loss_price,
    take_profit_price,
):
    # 计算交易数量
    risk = current_price - stop_loss_price
    reward = take_profit_price - current_price
    risk_reward_ratio = reward / risk
    win_probability = 0.6  # 交易获胜概率
    account_balance = 1000 / exchange_rate

    kelly_fraction = calculate_kelly(risk_reward_ratio, win_probability)
    trade_amount = account_balance * kelly_fraction

    minSz = product_info["minSz"]
    minSz = float(minSz)
    min_trade_size = trade_amount / current_price
    min_trade_size = round(min_trade_size, 5)
    return max(min_trade_size, minSz)


@sleep_and_retry
@limits(calls=60, period=2)
def place_order(
    ccy,
    side,
    size,
    current_price,
):
    order = tradeAPI.place_order(
        instId=ccy,
        tdMode="cash",
        side=side,
        ordType="limit",
        sz=str(size),
        px=str(current_price),
    )
    return order


@sleep_and_retry
@limits(calls=4, period=2)
def amend_order(ccy, ordId, newSz, newPx):
    order = tradeAPI.amend_order(ccy, ordId, newSz, newPx)
    return order


@sleep_and_retry
@limits(calls=60, period=2)
def monitor_order_status(ccy, order_id):
    # 获取订单信息 60次/2s
    order = tradeAPI.get_order(ccy, order_id)
    return order["data"][0]["state"]


@sleep_and_retry
@limits(calls=20, period=2)
def get_price_limit(ccy):
    # 获取限价 20次/2s
    response = publicAPI.get_price_limit(ccy)
    return response["data"][0]


@sleep_and_retry
@limits(calls=40, period=2)
def get_candlesticks(ccy, end_timestamp, bar):
    # 获取交易产品K线数据 40次/2s
    response = marketAPI.get_candlesticks(
        instId=ccy,
        before=str(end_timestamp),
        bar=bar,
        limit="300",
    )
    return response["data"]


@sleep_and_retry
@limits(calls=20, period=2)
def get_history_candlesticks(ccy, end_timestamp, bar):
    # 获取交易产品历史K线数据 20次/2s
    response = marketAPI.get_history_candlesticks(
        instId=ccy,
        after=str(end_timestamp),
        bar=bar,
        limit="100",
    )
    return response["data"]


def get_candles_paginated(ccy, bar, start_timestamp, end_timestamp):
    all_data = []

    while True:
        data = get_history_candlesticks(ccy, end_timestamp, bar=bar)
        if not data:
            break
        all_data.extend(data)

        if start_timestamp == int(data[0][0]):
            break
        end_timestamp = int(data[-1][0])
        time.sleep(0.01)

    return all_data


def get_price_data(ccy):
    # 创建一个 redis key
    cache_key = f"price_data:{ccy}"
    # 检查是否在 redis 存在
    cached_data = redis_client.get(cache_key)
    now = datetime.datetime.now()
    truncated_now = now.replace(second=0, microsecond=0)
    if cached_data:
        price_data = json.loads(cached_data)  # 转换为 list
        if len(price_data) > 0:
            # 根据上次数据确定开始时间
            start_timestamp = price_data[-1][0]
        else:
            start_time = truncated_now - datetime.timedelta(days=7)
            start_timestamp = int(start_time.timestamp() * 1000)

    else:
        start_time = truncated_now - datetime.timedelta(days=7)
        start_timestamp = int(start_time.timestamp() * 1000)
        price_data = []
    end_timestamp = int(truncated_now.timestamp() * 1000)

    # 获取历史数据
    historical_data = get_candles_paginated(ccy, "1m", start_timestamp, end_timestamp)

    # 获取最近数据
    recent_data = get_candlesticks(ccy, end_timestamp, "1m")

    # 合并数据
    all_data = historical_data + recent_data
    existing_timestamps = {entry[0] for entry in price_data}
    for candle in all_data:
        timestamp = int(candle[0])
        if timestamp not in existing_timestamps:
            close_price = float(candle[4])
            price_data.append([timestamp, close_price])

    # 根据时间戳排序
    price_data = sorted(price_data, key=lambda x: x[0])

    # 更新 redis
    redis_client.set(cache_key, json.dumps(price_data))

    return price_data


def predict_trend(price_data):
    """
    预测价格趋势
    price_data: 历史价格数据，二维数组，第一列是时间戳，第二列是价格
    """
    model = LinearRegression()

    # 将时间戳转换为特征矩阵
    X = np.array([data[0] for data in price_data]).reshape(-1, 1)
    y = np.array([data[1] for data in price_data])

    # 训练模型
    model.fit(X, y)

    # 预测未来的价格
    future_timestamp = X[-1][0] + 60000
    # 预测未来一分钟的价格，时间戳增加一分钟（以毫秒为单位）
    future_price = model.predict([[future_timestamp]])
    return round(future_price[0], 5)


@sleep_and_retry
@limits(calls=20, period=2)
def get_max_order_size(ccy, price):
    # 获取最大可买卖/开仓数量 20次/2s
    response = accountAPI.get_max_order_size(instId=ccy, tdMode="cash", px=str(price))
    return response["data"][0]


def check_balance(ccy, size, price):
    # 检查下单交易额度
    response = get_max_order_size(ccy, price)
    maxBuy = response["maxBuy"]
    maxBuy = float(maxBuy)
    return maxBuy >= size


@sleep_and_retry
@limits(calls=20, period=2)
def get_max_avail_size(ccy):
    # 获取最大可用数量 20次/2s
    info = accountAPI.get_max_avail_size(instId=ccy, tdMode="cash")
    return info["data"][0]


def auto_trade(symbols, stop_loss_price_ratio=0.02, take_profit_price_ratio=0.03):
    while True:
        try:
            for ccy in symbols:
                # 获取交易产品信息
                product_info = get_product_info(ccy)
                # 获取汇率信息
                exchange_rate = get_exchange_rate()
                console_log(ccy, "实时汇率", exchange_rate)

                # 获取当前价格
                buy_price = get_current_price(ccy)
                console_log(ccy, "实时价格", buy_price)

                # 止盈止损价格
                stop_loss_price = round(buy_price * (1 - stop_loss_price_ratio), 5)
                take_profit_price = round(buy_price * (1 + take_profit_price_ratio), 5)

                # 检查限价
                price_limit = get_price_limit(ccy)
                buyLmt = price_limit["buyLmt"]
                buyLmt = float(buyLmt)
                if take_profit_price > buyLmt:
                    take_profit_price = buyLmt

                # 计算交易数量
                size = calculate_trade_size(
                    product_info,
                    exchange_rate,
                    buy_price,
                    stop_loss_price,
                    take_profit_price,
                )
                console_log(ccy, "计算交易数量", size)

                # 检查账户余额是否足够支付订单
                if not check_balance(ccy, size, buy_price):
                    console_log(ccy, "账户余额不足买入", (size * buy_price))
                    break

                # 下单
                order = place_order(ccy, "buy", size, buy_price)
                if order["code"] != "0":
                    msg = order["msg"]
                    if order["code"] == "1":
                        msg = order["data"][0]["sMsg"]
                    console_log(ccy, "下单错误", msg)
                    break
                buy_order_id = order["data"][0]["ordId"]
                # 监控订单状态
                while True:
                    buy_status = monitor_order_status(ccy, buy_order_id)
                    if buy_status != "live":
                        console_log(ccy, "买入订单状态", buy_status)
                        break
                    else:
                        console_log(
                            ccy,
                            f"等待成交, 买入价格: {buy_price}, 当前价格: {get_current_price(ccy)}, 订单状态",
                            buy_status,
                        )
                if buy_status == "canceled":
                    continue

                # 监控止盈止损订单状态
                while True:
                    # 检查账户余额是否足够支付订单
                    avail = get_max_avail_size(ccy)
                    availSell = avail["availSell"]
                    availSell = float(availSell)
                    minSz = product_info["minSz"]
                    minSz = float(minSz)
                    if availSell < minSz:
                        console_log(ccy, "可用余额不足", availSell)
                        break

                    # 获取历史价格数据
                    price_data = get_price_data(ccy)

                    # 获取当前价格
                    current_price = get_current_price(ccy)

                    # 预测价格趋势
                    future_price = predict_trend(price_data)
                    # 根据预测趋势设置止盈止损
                    sell_order = None
                    if current_price >= take_profit_price:
                        sell_order = place_order(ccy, "sell", availSell, current_price)

                    if (
                        current_price <= stop_loss_price
                        or future_price < stop_loss_price
                    ):
                        sell_order = place_order(
                            ccy, "sell", availSell, stop_loss_price
                        )

                    if not sell_order:
                        console_log(
                            ccy,
                            f"当前价格 {current_price} 在 {stop_loss_price} ~ {take_profit_price} 范围内, 未来一分钟预测价格: {future_price}, 涨跌: ",
                            (round(current_price - buy_price, 5)),
                        )
                        continue

                    if sell_order["code"] != "0":
                        msg = sell_order["msg"]
                        if sell_order["code"] == "1":
                            msg = sell_order["data"][0]["sMsg"]
                        console_log(ccy, "下单止盈止损错误", msg)
                        break
                    sell_order_id = sell_order["data"][0]["ordId"]
                    status = monitor_order_status(ccy, sell_order_id)
                    if status != "live":
                        console_log(ccy, "止盈止损订单状态", status)
                        break
                    else:
                        console_log(ccy, "等待止盈止损成交, 订单状态", status)

        except Exception as e:
            logger.error(f"出现错误：{e}", traceback.format_exc())
            console_log(ccy, "出现错误", traceback.format_exc())
            time.sleep(5)


def console_log(ccy, target_name, target_value):
    print(
        f"币种: {ccy}",
        f"时间: {datetime.datetime.now()}",
        f"{target_name}: {target_value}",
    )


# 调用自动交易函数
if __name__ == "__main__":
    get_price_data("DOGE-USDT")
    # print("欢迎使用自动交易系统！正在初始化，请稍候...")
    # symbols = ["CEL-USDT"]  # 币种代码列表
    # auto_trade(symbols)
    # print("自动交易系统已停止运行")
