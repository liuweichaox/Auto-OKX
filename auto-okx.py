import datetime
import threading
import time
import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.PublicData as PublicData
from ratelimit import limits, sleep_and_retry

# 实盘交易设置
FLAG = "1"  # 实盘: 0 , 模拟盘：1
# API Key等信息
if FLAG == "0":
    BASE_URL = "https://www.okx.com"
    API_KEY = "f52b2961-8c08-4af4-876c-d4c6bcebdc6c"
    SECRET_KEY = "7DB206F3D875F9062170D14B1BC23BEF"
    PASSPHRASE = "Lwc1st+-"
elif FLAG == "1":
    BASE_URL = "https://www.okx.com"
    API_KEY = "12648afa-8e43-4d58-87f3-1a1510698ce2"
    SECRET_KEY = "D9A798DF9EBC04954835D887B577386F"
    PASSPHRASE = "Lwc1st+-"

PROXY = 'http://127.0.0.1:7890'

# 初始化API
accountAPI = Account.AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=FLAG, proxy=PROXY)
marketAPI = MarketData.MarketAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=FLAG, proxy=PROXY)
tradeAPI = Trade.TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG,debug=FLAG, proxy=PROXY)
publicAPI = PublicData.PublicAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG,debug=FLAG, proxy=PROXY)
# 选择币种列表
symbols = ["DOGE-USDT", "TON-USDT", "NOT-USDT"]  # 添加多个币种

def is_success(response):
    return response["code"] == "0"


def is_error(response):
    return response["code"] != "0"


def get_msg(response):
    return response["msg"]


def get_code(response):
    return response["code"]


def get_data(response, attribute):
    return response["data"][0][attribute]


# 获取USDT对CNY的汇率 1次/2s
@sleep_and_retry
@limits(calls=1, period=2)
def get_exchange_rate():
    response = marketAPI.get_exchange_rate()
    if is_error(response):
        raise ValueError(get_msg(response))
    result = float(get_data(response, "usdCny"))
    return result



# 获取交易产品基础信息 20次/2s
@sleep_and_retry
@limits(calls=20, period=2)
def get_ccy_min_size(ccy):
    result = publicAPI.get_instruments(instType='SPOT', instId=ccy)
    if is_error(result):
        raise ValueError(get_msg(result))
    result = float(get_data(result, 'minSz'))
    return result

# 计算最低下单金额
def get_min_trade_size(ccy):
    current_price = get_ticker(ccy)
    usdt_to_cny = get_exchange_rate()
    min_size = get_ccy_min_size(ccy)
    target_profit_usdt = 1 / usdt_to_cny  # 每 0.01% 比例盈利目标
    while round(min_size * current_price, 3) * 0.0001 < target_profit_usdt:
        min_size += 0.000001
    return round(min_size, 6)


# 下单函数 60次/2s
@sleep_and_retry
@limits(calls=60, period=2)
def place_order(ccy, side, size, price):
    order = tradeAPI.place_order(
        instId=ccy,
        tdMode="cash",
        side=side,
        ordType="limit",
        sz=str(size),
        px=str(price),
    )
    return order


# 获取当前价格 20次/2s
@sleep_and_retry
@limits(calls=20, period=2)
def get_ticker(ccy):
    ticker = marketAPI.get_ticker(ccy)
    if is_error(ticker):
        raise ValueError(get_msg(ticker))
    result = float(get_data(ticker, "last"))
    return result


def auto_place_order(ccy):
    current_price = get_ticker(ccy)
    min_trade_size = get_min_trade_size(ccy)
    if current_price * get_exchange_rate() > 100:
        return
    order = place_order(ccy, "buy", min_trade_size, current_price)
    if is_error(order):
        raise ValueError(get_msg(order))
    order_id = get_data(order, "ordId")
    buy_price = current_price
    print(
        f"买单已下单: {order_id}, 价格: {current_price}, 数量: {min_trade_size}"
    )
    sell_price = round(current_price * 1.0001, 3)  # 0.01% 目标收益
    stop_loss_price = round(current_price * 0.9992, 3)  # 0.0008% 止损比例
    print(f"设置卖出价格: {sell_price}, 止损价格: {stop_loss_price}")
    while True:
        current_price = get_ticker(ccy)
        if current_price >= sell_price:
            order = place_order(ccy, "sell", min_trade_size, current_price)
            if is_error(order):
                raise ValueError(get_msg(order))
            print(f"已卖出: {order}, 价格: {current_price}")
            break
        elif current_price <= stop_loss_price:
            order = place_order(ccy, "sell", min_trade_size, current_price)
            if is_error(order):
                raise ValueError(get_code(order))
            print(f"止损卖出: {order}, 价格: {current_price}")
            break
        print(f'{ccy}, 当前价格: {current_price}, 买入价: {buy_price}, 止盈价: {sell_price}, 止损价: {stop_loss_price}')
        time.sleep(5)

@sleep_and_retry
@limits(calls=10, period=2)
def foo():
    print(datetime.datetime.now())

if __name__ == "__main__":
    print('Auto OKX run')
    threads = []
    for ccy in symbols:
        thread = threading.Thread(target=auto_place_order, args=(ccy,))
        thread.start()
        threads.append(thread)
        time.sleep(5)

    for thread in threads:
        thread.join()
