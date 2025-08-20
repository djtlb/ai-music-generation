
class MockRedis:
    async def ping(self): return True
    async def close(self): pass

redis_client = MockRedis()

async def init_redis(): pass
async def close_redis(): pass
async def get_redis_client(): return redis_client
