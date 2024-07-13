import time
from api.okx_api import MarketAPI
import pandas as pd


def get_latest_candlestick_data(instId, bar):
    total_data = []
    after = ""
    while len(total_data) < 1440:
        data = MarketAPI.get_candlesticks(instId=instId, bar=bar, after=after)
        data = data["data"]
        if data is None or len(data) == 0:
            break
        total_data.extend(data)
        after = data[-1][0]
        time.sleep(0.1)
    return total_data


def get_price_data(symbol, timeframe):
    """
    获取指定交易对和时间范围的价格数据。

    参数:
    symbol (str): 交易对符号，例如"BTC-USDT"。
    timeframe (str): 时间范围，例如"1D"表示每日。

    返回:
    DataFrame: 包含价格数据的DataFrame。
    """
    data = get_latest_candlestick_data(symbol, timeframe)

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
