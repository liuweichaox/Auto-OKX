import time
import requests
import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade

# 实盘交易设置
FLAG = "1"  # 实盘: 0 , 模拟盘：1
# API Key等信息
if FLAG == "0":
    BASE_URL = "https://www.okx.com"
    API_KEY = "f52b2961-8c08-4af4-876c-d4c6bcebdc6c"
    SECRET_KEY = "7DB206F3D875F9062170D14B1BC23BEF"
    PASSPHRASE = ""
elif FLAG == "1":
    BASE_URL = "https://www.okx.com"
    API_KEY = "12648afa-8e43-4d58-87f3-1a1510698ce2"
    SECRET_KEY = "D9A798DF9EBC04954835D887B577386F"
    PASSPHRASE = ""

# 初始化API
accountAPI = Account.AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
marketAPI = MarketData.MarketAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
tradeAPI = Trade.TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)

# 选择币种列表
symbols = ["DOGE-USDT", "CORE-USDT", "CEL-USDT"]  # 添加多个币种


# 获取USDT对CNY的汇率
def get_usdt_to_cny_rate():
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USDT")
    rates = response.json()["rates"]
    cny_rate = rates["CNY"]
    return cny_rate


def get_ticker(symbol):
    response = marketAPI.get_ticker(symbol)
    if response['code']!='0':
        raise ValueError(response['msg'])
    return response["data"][0]


# 计算最低下单金额
def get_min_trade_amount(symbol):
    ticker = get_ticker(symbol)
    current_price = float(ticker["last"])
    usdt_to_cny = get_usdt_to_cny_rate()
    min_trade_amount = 1 / (current_price * usdt_to_cny * 0.001)  # 0.1% 目标收益
    return min_trade_amount, current_price


# 下单函数
def place_order(symbol, side, size, price):
    order = tradeAPI.place_order(
        instId=symbol,
        tdMode="cash",
        side=side,
        ordType="limit",
        sz=str(size),
        px=str(price),
    )
    return order


# 获取当前价格
def get_current_price(symbol):
    ticker = marketAPI.get_ticker(symbol)
    return float(ticker["data"][0]["last"])


# 主逻辑
def main():
    usdt_to_cny = get_usdt_to_cny_rate()
    target_profit_per_minute = 1  # 每分钟盈利目标
    target_profit_usdt = target_profit_per_minute / usdt_to_cny

    while True:
        for symbol in symbols:
            min_trade_amount, current_price = get_min_trade_amount(symbol)
            if current_price <= 100:
                order = place_order(symbol, "buy", min_trade_amount, current_price)
                code = order["code"]
                message = order["msg"]
                if code != "0":
                    print(f"下单失败：{message}")
                    continue
                order_id = order["data"][0]["ordId"]
                print(
                    f"买单已下单: {order_id}, 价格: {current_price}, 数量: {min_trade_amount}"
                )

                buy_price = current_price
                sell_price = buy_price * 1.001  # 0.1% 目标收益
                stop_loss_price = buy_price * 0.9995  # 0.05% 止损比例
                print(f"设置卖出价格: {sell_price}, 止损价格: {stop_loss_price}")

                while True:
                    current_price = get_current_price(symbol)
                    if current_price >= sell_price:
                        order = place_order(
                            symbol, "sell", min_trade_amount, current_price
                        )
                        print(f"已卖出: {order["data"][0]["ordId"]}, 价格: {current_price}")
                        break
                    elif current_price <= stop_loss_price:
                        order = place_order(
                            symbol, "sell", min_trade_amount, current_price
                        )
                        print(f"止损卖出: {order["data"][0]["ordId"]}, 价格: {current_price}")
                        break
                    time.sleep(5)
        time.sleep(60)  # 每分钟循环一次


if __name__ == "__main__":
    main()
