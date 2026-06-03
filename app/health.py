"""
GET /health — Feed freshness and DB connectivity check.
Responds within 200ms regardless of store count.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Response

from .database import get_pool
from .models import HealthResponse

logger = structlog.get_logger()
router = APIRouter()

STALE_THRESHOLD_MINUTES = 10


@router.get("/health", response_model=HealthResponse)
async def health_check(
    response: Response,
    pool: asyncpg.Pool = Depends(get_pool),
) -> HealthResponse:
    warnings: list[str] = []
    last_event_ts: dict[str, str | None] = {}

    try:
        async with asyncio.timeout(0.18):  # 180ms budget (stay under 200ms)
            async with pool.acquire() as conn:
                # Test DB connectivity
                await conn.fetchval("SELECT 1")

                # Get last event timestamp per store
                rows = await conn.fetch(
                    """
                    SELECT store_id, MAX(event_timestamp) AS last_ts
                    FROM entry_exit_events
                    GROUP BY store_id
                    """
                )

                now = datetime.now(tz=timezone.utc)
                for row in rows:
                    store_id = row["store_id"]
                    last_ts = row["last_ts"]
                    if last_ts:
                        last_ts_utc = last_ts.replace(tzinfo=timezone.utc) if last_ts.tzinfo is None else last_ts
                        last_event_ts[store_id] = last_ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                        age_minutes = (now - last_ts_utc).total_seconds() / 60
                        if age_minutes > STALE_THRESHOLD_MINUTES:
                            warnings.append(
                                f"STALE_FEED:{store_id} — last event {age_minutes:.1f}m ago (threshold {STALE_THRESHOLD_MINUTES}m)"
                            )
                    else:
                        last_event_ts[store_id] = None

        status = "ok"

    except asyncio.TimeoutError:
        logger.warning("health_check_timeout")
        response.status_code = 503
        return HealthResponse(
            status="degraded",
            last_event_timestamp_per_store={},
            warnings=["Health check timed out"],
        )
    except Exception as e:
        logger.critical("database_connection_lost", error=str(e))
        response.status_code = 503
        return HealthResponse(
            status="degraded",
            last_event_timestamp_per_store={},
            warnings=[f"Database unreachable: {str(e)}"],
        )

    return HealthResponse(
        status=status,
        last_event_timestamp_per_store=last_event_ts,
        warnings=warnings,
    )
