"""
GET /stores/{store_id}/heatmap — Zone heatmap API.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import asyncpg
import structlog
from fastapi import APIRouter, Depends

from .database import get_pool
from .models import HeatmapResponse, HeatmapZone

logger = structlog.get_logger()
router = APIRouter()

@router.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store_id: str,
    pool: asyncpg.Pool = Depends(get_pool),
) -> HeatmapResponse:
    today = date.today().isoformat()

    async with pool.acquire() as conn:
        # Get raw visits and unique sessions per zone
        visits_rows = await conn.fetch(
            """
            SELECT 
                zone_id,
                zone_name,
                COUNT(DISTINCT id_token) FILTER (WHERE event_type = 'zone_entered') AS raw_visits,
                COUNT(DISTINCT id_token) AS unique_sessions
            FROM zone_events
            WHERE store_id = $1
              AND DATE(event_time AT TIME ZONE 'UTC') = CURRENT_DATE
            GROUP BY zone_id, zone_name
            """,
            store_id,
        )

        # Get average dwell per zone (excluding staff)
        dwell_rows = await conn.fetch(
            """
            SELECT 
                zone_id, 
                AVG(dwell_ms) AS avg_dwell
            FROM zone_events
            WHERE store_id = $1
              AND event_type IN ('zone_exited', 'zone_dwell')
              AND is_staff = false
              AND dwell_ms IS NOT NULL
              AND DATE(event_time AT TIME ZONE 'UTC') = CURRENT_DATE
            GROUP BY zone_id
            """,
            store_id,
        )

    # Convert to dictionaries for easy lookup
    avg_dwell_map = {
        row["zone_id"]: float(row["avg_dwell"]) 
        for row in dwell_rows
    }

    # Find the maximum raw_visits for normalisation
    max_visits = max((row["raw_visits"] for row in visits_rows), default=0)

    zones = []
    for row in visits_rows:
        raw_visits = row["raw_visits"]
        # Normalise to 0-100 scale
        if max_visits > 0:
            visit_frequency = int((raw_visits / max_visits) * 100)
        else:
            visit_frequency = 0
            
        avg_dwell = avg_dwell_map.get(row["zone_id"])
        if avg_dwell is not None:
            avg_dwell = round(avg_dwell, 1)

        zones.append(
            HeatmapZone(
                zone_id=row["zone_id"],
                zone_name=row["zone_name"] or "Unknown",
                visit_frequency=visit_frequency,
                avg_dwell_ms=avg_dwell,
                data_confidence=(row["unique_sessions"] >= 20)
            )
        )

    return HeatmapResponse(
        store_id=store_id,
        zones=zones
    )
