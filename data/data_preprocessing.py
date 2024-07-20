import numpy as np
import ta


def feature_engineering(df):
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
    df["market_condition"] = df.apply(identify_market_condition, axis=1)

    df = calculate_macd(df)
    df = calculate_atr(df)
    df = calculate_adx(df)
    df = calculate_obv(df)
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_kdj(df)
    df.dropna(inplace=True)
    return df


def calculate_atr(df, window=14):
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


def calculate_macd(df):
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


def calculate_adx(df, window=14):
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


def calculate_obv(df):
    """
    计算累积成交量（OBV）指标。

    参数:
    df (DataFrame): 包含价格数据的DataFrame。

    返回:
    DataFrame: 包含OBV指标的DataFrame。
    """
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return df


def calculate_sma(df, window=5):
    """
    计算简单移动平均线（SMA）指标。

    参数:
    df (DataFrame): 包含价格数据的DataFrame。
    window (int): SMA计算窗口大小，默认为5。

    返回:
    DataFrame: 包含SMA指标的DataFrame。
    """
    df["sma"] = df["close"].rolling(window=window).mean()
    return df


def calculate_ema(df, window=5):
    """
    计算指数移动平均线（EMA）指标。

    参数:
    df (DataFrame): 包含价格数据的DataFrame。
    window (int): EMA计算窗口大小，默认为5。

    返回:
    DataFrame: 包含EMA指标的DataFrame。
    """
    df["ema"] = df["close"].ewm(span=window, adjust=False).mean()
    return df


def calculate_kdj(df, window=9):
    """
    计算KDJ指标。

    参数:
    df (DataFrame): 包含价格数据的DataFrame。
    window (int): KDJ计算窗口大小，默认为9。

    返回:
    DataFrame: 包含KDJ指标的DataFrame。
    """
    low_list = df["low"].rolling(window=window).min()
    low_list.fillna(value=df["low"].expanding().min(), inplace=True)
    high_list = df["high"].rolling(window=window).max()
    high_list.fillna(value=df["high"].expanding().max(), inplace=True)

    df["rsv"] = (df["close"] - low_list) / (high_list - low_list) * 100
    df["k"] = df["rsv"].ewm(com=2).mean()
    df["d"] = df["k"].ewm(com=2).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    return df


def identify_market_condition(row):
    if row["sma"] > row["ema"]:
        if row["close"] > row["sma"]:
            return "Uptrend (Single-directional)"
        else:
            return "Downtrend (Single-directional)"
    if row["close"] < row["bb_high"] and row["close"] > row["bb_low"]:
        return "Range-bound (Oscillating)"
    return "Unknown"


def fetch_feature(price_data):
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
            "obv",
            "sma",
            "ema",
            "rsv",
            "k",
            "d",
            "j",
        ]
    ]
    y = price_data["close"]
    return X, y
