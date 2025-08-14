"""
Redis client configuration and connection management
"""

import aioredis
from config.settings import get_settings
import logging
import json
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global Redis client
redis_client = None

async def get_redis_client():
    """Get Redis client instance"""
    global redis_client
    
    if redis_client is None:
        settings = get_settings()
        redis_client = aioredis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20
        )
        logger.info("Redis client initialized")
    
    return redis_client

class RedisCache:
    """Redis cache wrapper with common operations"""
    
    def __init__(self):
        self.client = None
    
    async def init(self):
        """Initialize Redis connection"""
        self.client = await get_redis_client()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            await self.init()
        
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        """Set value in cache with optional expiration"""
        if not self.client:
            await self.init()
        
        try:
            serialized_value = json.dumps(value, default=str)
            await self.client.set(key, serialized_value, ex=expire)
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {str(e)}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.client:
            await self.init()
        
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {str(e)}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.client:
            await self.init()
        
        try:
            return bool(await self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {str(e)}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment key value"""
        if not self.client:
            await self.init()
        
        try:
            return await self.client.incr(key, amount)
        except Exception as e:
            logger.error(f"Redis INCR error for key {key}: {str(e)}")
            return 0
    
    async def set_hash(self, key: str, mapping: dict):
        """Set hash values"""
        if not self.client:
            await self.init()
        
        try:
            await self.client.hset(key, mapping=mapping)
        except Exception as e:
            logger.error(f"Redis HSET error for key {key}: {str(e)}")
    
    async def get_hash(self, key: str, field: str) -> Optional[str]:
        """Get hash field value"""
        if not self.client:
            await self.init()
        
        try:
            return await self.client.hget(key, field)
        except Exception as e:
            logger.error(f"Redis HGET error for key {key}, field {field}: {str(e)}")
            return None

# Global cache instance
cache = RedisCache()
