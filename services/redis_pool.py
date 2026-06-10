import os
import asyncio
from urllib.parse import urlparse

REDIS_URL = os.environ.get("REDIS_URL", "")
_pool = None
_max_retries = 3
_retry_delay = 1.0


def _build_settings():
    from arq.connections import RedisSettings
    if REDIS_URL.startswith(("redis://", "rediss://", "unix://")):
        return RedisSettings.from_dsn(REDIS_URL, max_connections=10, retry_on_timeout=True, socket_keepalive=True)
    parsed = urlparse(REDIS_URL)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password or None,
        max_connections=10,
        retry_on_timeout=True,
        socket_keepalive=True,
    )


async def get_redis_pool():
    global _pool
    if _pool is None:
        if not REDIS_URL:
            return None
        _pool = await _connect_with_retry()
    return _pool


async def _connect_with_retry():
    from arq.connections import create_pool
    last_exc = None
    for attempt in range(1, _max_retries + 1):
        try:
            settings = _build_settings()
            pool = await create_pool(settings)
            print(f"[REDIS] Connected (attempt {attempt})")
            return pool
        except Exception as e:
            last_exc = e
            print(f"[REDIS] Connection attempt {attempt}/{_max_retries} failed: {e}")
            if attempt < _max_retries:
                await asyncio.sleep(_retry_delay * (2 ** (attempt - 1)))
    print(f"[REDIS] All {_max_retries} attempts failed: {last_exc}")
    return None


async def reconnect():
    global _pool
    if _pool:
        try:
            _pool.close()
            await _pool.wait_closed()
        except Exception:
            pass
        _pool = None
    _pool = await _connect_with_retry()
    return _pool


async def close_redis_pool():
    global _pool
    if _pool:
        try:
            _pool.close()
            await _pool.wait_closed()
        except Exception:
            pass
        _pool = None


def redis_available() -> bool:
    return bool(REDIS_URL)
