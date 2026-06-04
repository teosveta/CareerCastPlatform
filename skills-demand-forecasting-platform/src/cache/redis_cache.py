"""
Redis-based caching system for improved performance.
Provides intelligent caching with TTL management and cache warming.
"""

import json
import logging
from typing import Any, Optional, Dict, List
import hashlib
from datetime import datetime, timedelta
import pickle

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Advanced Redis caching system with intelligent cache management,
    TTL optimization, and cache warming capabilities.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = None
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0}

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host, port=port, db=db,
                    decode_responses=False,  # Handle binary data
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.redis_client = None
        else:
            logger.warning("Redis library not installed. Caching disabled.")

    def _generate_key(self, key: str, namespace: str = "skills_platform") -> str:
        """Generate namespaced cache key."""
        return f"{namespace}:{key}"

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage."""
        try:
            # Try JSON first for simple data
            json_str = json.dumps(data, default=str)
            return json_str.encode('utf-8')
        except (TypeError, ValueError):
            # Use pickle for complex objects
            return pickle.dumps(data)

    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Use pickle
            return pickle.loads(data)

    async def get(self, key: str, namespace: str = "skills_platform") -> Optional[Any]:
        """Get value from cache."""
        if not self.redis_client:
            return None

        try:
            cache_key = self._generate_key(key, namespace)
            data = self.redis_client.get(cache_key)

            if data:
                self.cache_stats["hits"] += 1
                return self._deserialize_data(data)
            else:
                self.cache_stats["misses"] += 1
                return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["misses"] += 1
            return None

    async def set(
            self,
            key: str,
            value: Any,
            ttl: int = 3600,
            namespace: str = "skills_platform"
    ) -> bool:
        """Set value in cache with TTL."""
        if not self.redis_client:
            return False

        try:
            cache_key = self._generate_key(key, namespace)
            serialized_data = self._serialize_data(value)

            result = self.redis_client.setex(cache_key, ttl, serialized_data)

            if result:
                self.cache_stats["sets"] += 1
                return True
            return False

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str, namespace: str = "skills_platform") -> bool:
        """Delete key from cache."""
        if not self.redis_client:
            return False

        try:
            cache_key = self._generate_key(key, namespace)
            result = self.redis_client.delete(cache_key)
            return bool(result)

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def exists(self, key: str, namespace: str = "skills_platform") -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False

        try:
            cache_key = self._generate_key(key, namespace)
            return bool(self.redis_client.exists(cache_key))

        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False

    async def get_multiple(self, keys: List[str], namespace: str = "skills_platform") -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.redis_client:
            return {}

        try:
            cache_keys = [self._generate_key(key, namespace) for key in keys]
            values = self.redis_client.mget(cache_keys)

            result = {}
            for i, value in enumerate(values):
                if value:
                    result[keys[i]] = self._deserialize_data(value)
                    self.cache_stats["hits"] += 1
                else:
                    self.cache_stats["misses"] += 1

            return result

        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            return {}

    async def set_multiple(
            self,
            data: Dict[str, Any],
            ttl: int = 3600,
            namespace: str = "skills_platform"
    ) -> bool:
        """Set multiple values in cache."""
        if not self.redis_client:
            return False

        try:
            pipe = self.redis_client.pipeline()

            for key, value in data.items():
                cache_key = self._generate_key(key, namespace)
                serialized_data = self._serialize_data(value)
                pipe.setex(cache_key, ttl, serialized_data)

            results = pipe.execute()

            success_count = sum(1 for result in results if result)
            self.cache_stats["sets"] += success_count

            return success_count == len(data)

        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_ratio = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        stats = {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "sets": self.cache_stats["sets"],
            "hit_ratio_percentage": round(hit_ratio, 2),
            "total_requests": total_requests
        }

        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_memory_used": info.get("used_memory_human", "Unknown"),
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_keyspace_hits": info.get("keyspace_hits", 0),
                    "redis_keyspace_misses": info.get("keyspace_misses", 0)
                })
            except Exception as e:
                logger.warning(f"Could not get Redis info: {e}")

        return stats

    def flush_cache(self, namespace: Optional[str] = None) -> bool:
        """Flush cache (optionally by namespace)."""
        if not self.redis_client:
            return False

        try:
            if namespace:
                # Delete keys by pattern
                pattern = f"{namespace}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                # Flush all
                self.redis_client.flushdb()

            return True

        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False

    async def warm_cache(self, warming_data: Dict[str, Any]) -> None:
        """Warm cache with commonly accessed data."""
        if not self.redis_client:
            return

        logger.info("Warming cache with frequently accessed data...")

        try:
            # Set with longer TTL for warm data
            await self.set_multiple(warming_data, ttl=7200)  # 2 hours
            logger.info(f"Cache warmed with {len(warming_data)} items")

        except Exception as e:
            logger.error(f"Cache warming failed: {e}")