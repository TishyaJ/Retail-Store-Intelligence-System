"""
GET /stores/{store_id}/metrics — Real-time store metrics.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import asyncpg
import structlog
from fastapi import APIRouter, Depends

from .database import get_pool
from .models import StoreMetricsResponse

logger = structlog.get_logger()
router = APIRouter()


@router.get("/stores/{store_id}/metrics", response_model=StoreMetricsResponse)
async def get_metrics(
    store_id: str,
    pool: asyncpg.Pool = Depends(get_pool),
) -> StoreMetricsResponse:
    today = date.today().isoformat()

    async with pool.acquire() as conn:
        # Unique visitors: distinct id_tokens from entry events today
        unique_visitors_row = await conn.fetchrow(
            """
            SELECT COUNT(DISTINCT id_token) AS cnt
            FROM entry_exit_events
            WHERE store_id = $1
              AND event_type = 'entry'
              AND is_staff = false
              AND DATE(event_timestamp AT TIME ZONE 'UTC') = CURRENT_DATE
            """,
            store_id,
        )
        unique_visitors = int(unique_visitors_row["cnt"] or 0)

        # Conversion rate: visitors who were in billing zone within 5 min of POS txn
        conversion_count = 0
        if unique_visitors > 0:
            conversion_row = await conn.fetchrow(
                """
                SELECT COUNT(DISTINCT qe.id_token) AS cnt
                FROM queue_events qe
                JOIN (
                    SELECT store_id,
                           (order_date + order_time)::TIMESTAMPTZ AS order_ts
                    FROM pos_transactions
                    WHERE store_id = $1
                      AND order_date = CURRENT_DATE
                    GROUP BY store_id, order_date, order_time
                ) txn ON txn.store_id = qe.store_id
                      AND qe.queue_exit_ts BETWEEN txn.order_ts - INTERVAL '5 minutes'
                                                AND txn.order_ts + INTERVAL '1 minute'
                WHERE qe.store_id = $1
                  AND DATE(qe.queue_join_ts AT TIME ZONE 'UTC') = CURRENT_DATE
                  AND qe.abandoned = false
                  AND qe.id_token IS NOT NULL
                """,
                store_id,
            )
            conversion_count = int(conversion_row["cnt"] or 0)

        conversion_rate = (
            round(conversion_count / unique_visitors, 4) if unique_visitors > 0 else 0.0
        )

        # Average dwell per zone
        dwell_rows = await conn.fetch(
            """
            SELECT zone_id, AVG(dwell_ms) AS avg_dwell
            FROM zone_events
            WHERE store_id = $1
              AND event_type IN ('zone_exited', 'zone_dwell')
              AND is_staff = false
              AND DATE(event_time AT TIME ZONE 'UTC') = CURRENT_DATE
              AND dwell_ms IS NOT NULL
            GROUP BY zone_id
            """,
            store_id,
        )
        avg_dwell_per_zone = {
            row["zone_id"]: round(float(row["avg_dwell"]), 1)
            for row in dwell_rows
        }

        # Current queue depth: visitors in billing zone not yet exited
        queue_row = await conn.fetchrow(
            """
            SELECT COUNT(*) AS cnt
            FROM queue_events
            WHERE store_id = $1
              AND DATE(queue_join_ts AT TIME ZONE 'UTC') = CURRENT_DATE
              AND queue_exit_ts > NOW()
            """,
            store_id,
        )
        queue_depth = int(queue_row["cnt"] or 0)

        # Abandonment rate
        abandon_row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE abandoned = true) AS abandoned_cnt,
                COUNT(*) AS total_cnt
            FROM queue_events
            WHERE store_id = $1
              AND DATE(queue_join_ts AT TIME ZONE 'UTC') = CURRENT_DATE
            """,
            store_id,
        )
        total_q = int(abandon_row["total_cnt"] or 0)
        abandoned_q = int(abandon_row["abandoned_cnt"] or 0)
        abandonment_rate = round(abandoned_q / total_q, 4) if total_q > 0 else 0.0

    logger.info("metrics_computed", store_id=store_id, unique_visitors=unique_visitors)

    return StoreMetricsResponse(
        store_id=store_id,
        date=today,
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_per_zone=avg_dwell_per_zone,
        queue_depth=queue_depth,
        abandonment_rate=abandonment_rate,
    )
