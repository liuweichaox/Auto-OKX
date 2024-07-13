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

    df = calculate_macd(df)
    df = calculate_atr(df)
    df = calculate_adx(df)
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
        ]
    ]
    y = price_data["close"]
    return X, y
