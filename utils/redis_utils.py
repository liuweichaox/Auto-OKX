import redis


RedisClient = redis.StrictRedis(host="localhost", port=6379, db=0, password="123456")
