import redis
import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.PublicData as PublicData
import okx.Funding as Funding


class FakeConfig:
    FLAG = "1"
    DEBUG = True
    BASE_URL = "https://www.okx.com"
    API_KEY = "12648afa-8e43-4d58-87f3-1a1510698ce2"
    SECRET_KEY = "D9A798DF9EBC04954835D887B577386F"
    PASSPHRASE = "Lwc1st+-"
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_PASSWORD = "123456"
    REDIS_DB = 1
