import os
from urllib.parse import urlparse

REDIS_URL = os.environ.get("REDIS_URL", "")
_pool = None


async def get_redis_pool():
    global _pool
    if _pool is None and REDIS_URL:
        from arq.connections import RedisSettings, create_pool

        if REDIS_URL.startswith(("redis://", "rediss://", "unix://")):
            settings = RedisSettings.from_dsn(REDIS_URL)
        else:
            parsed = urlparse(REDIS_URL)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379
            password = parsed.password or None
            settings = RedisSettings(host=host, port=port, password=password)

        _pool = await create_pool(settings)
    return _pool


async def close_redis_pool():
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


def redis_available() -> bool:
    return bool(REDIS_URL)
