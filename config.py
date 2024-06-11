import redis
import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.PublicData as PublicData
import okx.Funding as Funding


class Config:
    FLAG = "0"
    DEBUG = True
    BASE_URL = "https://www.okx.com"
    API_KEY = "f52b2961-8c08-4af4-876c-d4c6bcebdc6c"
    SECRET_KEY = "7DB206F3D875F9062170D14B1BC23BEF"
    PASSPHRASE = "Lwc1st+-"
    REDIS_HOST = "localhost"
    REIDS_PORT = 6379
    REDIS_PASSWORD = "123456"
    REDIS_DB = 0

    REDIS_CLIENT = redis.StrictRedis(
        host="localhost", port=6379, db=0, password="123456"
    )
    ACCOUNT_API = Account.AccountAPI(
        API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=DEBUG
    )
    MARKET_API = MarketData.MarketAPI(
        API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=DEBUG
    )
    TRADE_API = Trade.TradeAPI(
        API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=DEBUG
    )
    PUBLIC_API = PublicData.PublicAPI(
        API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=DEBUG
    )
    FUNDING_API = Funding.FundingAPI(
        API_KEY, SECRET_KEY, PASSPHRASE, False, flag=FLAG, debug=DEBUG
    )
