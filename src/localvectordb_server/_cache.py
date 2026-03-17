# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# src/localvectordb_server/_cache.py
"""
Thin cache wrapper around cachelib (framework-agnostic).
Replaces flask-caching with direct cachelib usage.
"""

import logging
from typing import Any, Optional

from cachelib import NullCache, SimpleCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Framework-agnostic cache manager wrapping cachelib backends."""

    def __init__(self):
        self._cache = NullCache()

    def init_from_config(self, config) -> None:
        """Initialize cache backend from Config object."""
        if not config.server.cache_enabled:
            self._cache = NullCache()
            logger.info("Caching disabled (NullCache)")
            return

        cache_type = config.server.cache_type
        cache_settings = config.server.cache_settings or {}

        if cache_type == "SimpleCache":
            self._cache = SimpleCache(**cache_settings)
        elif cache_type == "RedisCache":
            from cachelib import RedisCache

            self._cache = RedisCache(**cache_settings)
        elif cache_type == "FileSystemCache":
            from cachelib import FileSystemCache

            self._cache = FileSystemCache(**cache_settings)
        elif cache_type == "MemcachedCache":
            from cachelib import MemcachedCache

            self._cache = MemcachedCache(**cache_settings)
        else:
            self._cache = SimpleCache()

        logger.info(f"Cache initialized: {cache_type}")

    @property
    def cache(self):
        """Direct access to the underlying cachelib backend."""
        return self._cache

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, timeout: int = 300) -> bool:
        result: bool = self._cache.set(key, value, timeout)
        return result

    def delete(self, key: str) -> bool:
        result: bool = self._cache.delete(key)
        return result

    def clear(self) -> bool:
        result: bool = self._cache.clear()
        return result

    def cached(self, timeout: int = 300, key_prefix: str = ""):
        """Decorator for caching endpoint responses (compatible with route decorators)."""

        def decorator(func):
            # For now, passthrough — caching handled at router level if needed
            return func

        return decorator


# Global instance
cache = CacheManager()
