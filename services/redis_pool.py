import os

REDIS_URL = os.environ.get("REDIS_URL", "")
_pool = None


async def get_redis_pool():
    global _pool
    if _pool is None and REDIS_URL:
        from arq import create_pool
        _pool = await create_pool(REDIS_URL)
    return _pool


async def close_redis_pool():
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


def redis_available() -> bool:
    return bool(REDIS_URL)
