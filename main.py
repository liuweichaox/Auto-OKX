from sklearn.model_selection import train_test_split
from utils.redis_utils import RedisClient
from data.data_fetcher import get_price_data
import pandas as pd
import numpy as np
import tensorflow as tf

data = get_price_data("BTC-USDT", "5m")
# 归一化数据
data["price"] = (data["price"] - data["price"].min()) / (
    data["price"].max() - data["price"].min()
)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(
    data[["timestamp", "open", "high", "low", "volume"]], data["price"], test_size=0.2
)

# 创建模型
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

# 编译模型
model.compile(optimizer="adam", loss="mse")

# 训练模型
model.fit(X_train, y_train, epochs=100)

# 评估模型
score = model.evaluate(X_test, y_test)
