from models.train_deep_learning_model import train_models
from data.data_fetcher import get_price_data
from data.data_preprocessing import feature_engineering, identify_market_condition

price_data = get_price_data("BTC-USDT", "5m")

price_data = feature_engineering(price_data)

print(identify_market_condition(price_data))

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

lr_model, rf_model = train_models(X, y)
