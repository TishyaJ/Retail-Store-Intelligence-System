"""
GET /stores/{store_id}/funnel  — Conversion funnel with POS correlation.
POST /admin/pos/reload         — Reload POS_transactions.csv into DB.
"""
from __future__ import annotations

import os
from datetime import date

import asyncpg
import pandas as pd
import structlog
from fastapi import APIRouter, Depends, HTTPException

from .database import get_pool
from .models import FunnelResponse, FunnelStage, HeatmapResponse, HeatmapZone

logger = structlog.get_logger()
router = APIRouter()


def _drop_pct(current: int, previous: int) -> float:
    if previous == 0:
        return 0.0
    return round((1 - current / previous) * 100, 2)


@router.get("/stores/{store_id}/funnel", response_model=FunnelResponse)
async def get_funnel(
    store_id: str,
    pool: asyncpg.Pool = Depends(get_pool),
) -> FunnelResponse:
    today = date.today().isoformat()

    async with pool.acquire() as conn:
        # Stage 1: entry_count — distinct id_tokens from entry events today (excl staff, dedup re-entries)
        entry_row = await conn.fetchrow(
            """
            SELECT COUNT(DISTINCT id_token) AS cnt
            FROM entry_exit_events
            WHERE store_id = $1
              AND event_type IN ('entry', 'reentry')
              AND is_staff = false
              AND DATE(event_timestamp AT TIME ZONE 'UTC') = CURRENT_DATE
            """,
            store_id,
        )
        entry_count = int(entry_row["cnt"] or 0)

        # Stage 2: zone_visit_count — distinct id_tokens with at least one zone_entered today
        # Links via id_token (pipeline sets id_token on zone events when linked)
        zone_row = await conn.fetchrow(
            """
            SELECT COUNT(DISTINCT id_token) AS cnt
            FROM zone_events
            WHERE store_id = $1
              AND event_type = 'zone_entered'
              AND is_staff = false
              AND id_token IS NOT NULL
              AND DATE(event_time AT TIME ZONE 'UTC') = CURRENT_DATE
            """,
            store_id,
        )
        zone_visit_count = int(zone_row["cnt"] or 0)

        # Stage 3: billing_queue_count — distinct id_tokens in queue events today
        billing_row = await conn.fetchrow(
            """
            SELECT COUNT(DISTINCT id_token) AS cnt
            FROM queue_events
            WHERE store_id = $1
              AND id_token IS NOT NULL
              AND DATE(queue_join_ts AT TIME ZONE 'UTC') = CURRENT_DATE
            """,
            store_id,
        )
        billing_queue_count = int(billing_row["cnt"] or 0)

        # Stage 4: purchase_count — visitors correlated with POS txn via 5-min window
        # Key: use queue_exit_ts (when visitor LEFT billing zone) ± 5 min of order_ts
        purchase_row = await conn.fetchrow(
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
        purchase_count = int(purchase_row["cnt"] or 0)

    stages = [
        FunnelStage(stage="entry",         count=entry_count,         drop_off_pct=0.0),
        FunnelStage(stage="zone_visit",    count=zone_visit_count,    drop_off_pct=_drop_pct(zone_visit_count, entry_count)),
        FunnelStage(stage="billing_queue", count=billing_queue_count, drop_off_pct=_drop_pct(billing_queue_count, zone_visit_count)),
        FunnelStage(stage="purchase",      count=purchase_count,      drop_off_pct=_drop_pct(purchase_count, billing_queue_count)),
    ]

    logger.info("funnel_computed", store_id=store_id, entry=entry_count, purchase=purchase_count)
    return FunnelResponse(store_id=store_id, date=today, stages=stages)


@router.post("/admin/pos/reload")
async def reload_pos(pool: asyncpg.Pool = Depends(get_pool)) -> dict:
    """Reload POS_transactions.csv into the pos_transactions table."""
    csv_path = os.environ.get("POS_CSV_PATH", "POS_transactions.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        # Parse DD-MM-YYYY date format
        df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y").dt.date
        df["order_time"] = pd.to_datetime(df["order_time"], format="%H:%M:%S").dt.time
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)

        rows_inserted = 0
        async with pool.acquire() as conn:
            for _, row in df.iterrows():
                await conn.execute(
                    """
                    INSERT INTO pos_transactions
                      (order_id, order_date, order_time, store_id, product_id, brand_name, total_amount)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (order_id) DO NOTHING
                    """,
                    int(row["order_id"]),
                    row["order_date"],
                    row["order_time"],
                    str(row["store_id"]),
                    int(row["product_id"]),
                    str(row.get("brand_name", "")),
                    float(row["total_amount"]),
                )
                rows_inserted += 1

        logger.info("pos_reload_complete", rows=rows_inserted, csv_path=csv_path)
        return {"status": "ok", "rows_processed": rows_inserted}

    except Exception as e:
        logger.error("pos_reload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store_id: str,
    pool: asyncpg.Pool = Depends(get_pool),
) -> HeatmapResponse:
    """Zone visit frequency (normalised 0–100) and avg dwell times."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                zone_id,
                zone_name,
                COUNT(DISTINCT id_token) FILTER (WHERE event_type = 'zone_entered') AS visit_count,
                AVG(dwell_ms) FILTER (WHERE event_type IN ('zone_exited', 'zone_dwell')
                                      AND is_staff = false
                                      AND dwell_ms IS NOT NULL) AS avg_dwell
            FROM zone_events
            WHERE store_id = $1
              AND DATE(event_time AT TIME ZONE 'UTC') = CURRENT_DATE
            GROUP BY zone_id, zone_name
            """,
            store_id,
        )

    if not rows:
        return HeatmapResponse(store_id=store_id, zones=[])

    max_visits = max((int(r["visit_count"] or 0) for r in rows), default=1) or 1
    DATA_CONFIDENCE_THRESHOLD = 20

    zones = []
    for row in rows:
        visit_count = int(row["visit_count"] or 0)
        zones.append(HeatmapZone(
            zone_id=row["zone_id"],
            zone_name=row["zone_name"],
            visit_frequency=round(visit_count / max_visits * 100),
            avg_dwell_ms=round(float(row["avg_dwell"]), 1) if row["avg_dwell"] else None,
            data_confidence=visit_count >= DATA_CONFIDENCE_THRESHOLD,
        ))

    return HeatmapResponse(store_id=store_id, zones=zones)

