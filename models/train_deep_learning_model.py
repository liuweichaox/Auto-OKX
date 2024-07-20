from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.nlp_utils import sentiment_analysis


def train_models(X, y):
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


# 预测未来收盘价格
def predict_future_close(lr_model, rf_model, price_data):

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
            "obv",
            "sma",
            "ema",
            "rsv",
            "k",
            "d",
            "j",
        ]
    ].values.reshape(1, -1)

    lr_pred = lr_model.predict(latest_features)[0]
    rf_pred = rf_model.predict(latest_features)[0]

    sentiment_text = "Bitcoin price is soaring due to increased institutional interest."
    sentiment_score = sentiment_analysis(sentiment_text)
    avg_pred = combined_prediction(lr_pred, rf_pred, sentiment_score)

    return avg_pred


def combined_prediction(lr_pred, rf_pred, sentiment_score):
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
