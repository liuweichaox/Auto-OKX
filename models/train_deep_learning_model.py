from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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
