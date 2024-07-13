from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_analysis(text):
    """
    对文本进行情感分析。

    参数:
    text (str): 要分析的文本。

    返回:
    float: 情感分数。
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]
