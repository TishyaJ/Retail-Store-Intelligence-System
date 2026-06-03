"""
Database connection pool management using asyncpg.
"""
from __future__ import annotations

import os
from typing import Optional

import asyncpg
import structlog

logger = structlog.get_logger()

_pool: Optional[asyncpg.Pool] = None


async def create_pool() -> asyncpg.Pool:
    global _pool
    db_url = os.environ["DATABASE_URL"]
    # asyncpg expects postgresql:// not postgresql+asyncpg://
    dsn = db_url.replace("postgresql+asyncpg://", "postgresql://")
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    logger.info("database_pool_created", dsn=dsn.split("@")[-1])
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        logger.info("database_pool_closed")
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised")
    return _pool
