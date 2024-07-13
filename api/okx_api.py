import time
import logging
import redis
import time
import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.PublicData as PublicData
import okx.Funding as Funding
from ratelimit import limits, sleep_and_retry
import settings

api_config = {
    "api_key": settings.API_KEY,
    "api_secret_key": settings.SECRET_KEY,
    "passphrase": settings.PASSPHRASE,
    "use_server_time": False,
    "domain": settings.BASE_URL,
    "debug": settings.DEBUG,
    "proxy": None,
    "flag": settings.FLAG,
}
AccountAPI = Account.AccountAPI(**api_config)
MarketAPI = MarketData.MarketAPI(**api_config)
TradeAPI = Trade.TradeAPI(**api_config)
PublicAPI = PublicData.PublicAPI(**api_config)
FundingAPI = Funding.FundingAPI(**api_config)
