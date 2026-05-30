"""ARQ worker for processing document extraction jobs.

Run with:
  python worker.py
or
  arq worker.worker.WorkerSettings
"""

import os
from dotenv import load_dotenv

load_dotenv()

from services.task_queue import process_document_job

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


class WorkerSettings:
    functions = [process_document_job]
    redis_settings = REDIS_URL


if __name__ == "__main__":
    import asyncio
    from arq.connections import RedisSettings, create_pool
    from arq.worker import Worker

    async def main():
        settings = RedisSettings.from_dsn(REDIS_URL)
        redis = await create_pool(settings)
        worker = Worker(redis, functions=[process_document_job])
        print(f"[WORKER] ARQ worker started — connected to {REDIS_URL}")
        print(f"[WORKER] Registered functions: {[f.__name__ for f in [process_document_job]]}")
        await worker.run()

    asyncio.run(main())
