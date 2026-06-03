"""
GET /stores/{store_id}/anomalies — Anomaly detection.

Algorithms:
  - Queue spike: EWMA (alpha=0.3). WARN >50%, CRITICAL >100%
  - Conversion drop: Z-Score over 7-day rolling window. WARN if Z < -2.0
  - Dead zone: No ZONE_ENTER in last 4 hours during store hours. INFO
"""
from __future__ import annotations

import math
from datetime import date

import asyncpg
import structlog
from fastapi import APIRouter, Depends

from .database import get_pool
from .models import AnomalyItem, AnomalyResponse

logger = structlog.get_logger()
router = APIRouter()


@router.get("/stores/{store_id}/anomalies", response_model=AnomalyResponse)
async def get_anomalies(
    store_id: str,
    pool: asyncpg.Pool = Depends(get_pool),
) -> AnomalyResponse:
    anomalies: list[AnomalyItem] = []

    async with pool.acquire() as conn:
        # ----------------------------------------------------------------
        # 1. QUEUE SPIKE — EWMA on queue_position_at_join values
        # ----------------------------------------------------------------
        queue_rows = await conn.fetch(
            """
            SELECT queue_position_at_join
            FROM queue_events
            WHERE store_id = $1
              AND DATE(queue_join_ts AT TIME ZONE 'UTC') = CURRENT_DATE
            ORDER BY queue_join_ts
            """,
            store_id,
        )

        if len(queue_rows) >= 2:
            alpha = 0.3
            positions = [r["queue_position_at_join"] for r in queue_rows]
            ewma = float(positions[0])
            for p in positions[1:-1]:
                ewma = alpha * p + (1 - alpha) * ewma
            current = float(positions[-1])

            if ewma > 0:
                if current > 2.0 * ewma:
                    anomalies.append(AnomalyItem(
                        anomaly_type="QUEUE_SPIKE",
                        severity="CRITICAL",
                        description=f"Queue position {current:.0f} exceeds 7-day avg {ewma:.1f} by {((current/ewma-1)*100):.0f}%",
                        suggested_action="Open additional billing counter immediately",
                    ))
                elif current > 1.5 * ewma:
                    anomalies.append(AnomalyItem(
                        anomaly_type="QUEUE_SPIKE",
                        severity="WARN",
                        description=f"Queue position {current:.0f} is {((current/ewma-1)*100):.0f}% above smoothed avg {ewma:.1f}",
                        suggested_action="Monitor billing queue and prepare to open additional counter",
                    ))

        # ----------------------------------------------------------------
        # 2. CONVERSION DROP — Z-Score over 7-day rolling window
        # ----------------------------------------------------------------
        hist_rows = await conn.fetch(
            """
            SELECT
                DATE(e.event_timestamp AT TIME ZONE 'UTC') AS day,
                COUNT(DISTINCT e.id_token)::float AS entries,
                COUNT(DISTINCT q.id_token)::float AS purchases
            FROM entry_exit_events e
            LEFT JOIN queue_events q
                ON q.store_id = e.store_id
               AND q.id_token = e.id_token
               AND q.abandoned = false
               AND DATE(q.queue_join_ts AT TIME ZONE 'UTC') = DATE(e.event_timestamp AT TIME ZONE 'UTC')
            WHERE e.store_id = $1
              AND e.event_type = 'entry'
              AND e.is_staff = false
              AND DATE(e.event_timestamp AT TIME ZONE 'UTC') >= CURRENT_DATE - 7
            GROUP BY day
            ORDER BY day
            """,
            store_id,
        )

        if len(hist_rows) >= 3:
            rates = [
                (float(r["purchases"]) / float(r["entries"]) if float(r["entries"]) > 0 else 0.0)
                for r in hist_rows
            ]
            # Exclude today (last row) from historical average
            hist_rates = rates[:-1]
            current_rate = rates[-1]
            if len(hist_rates) >= 2:
                mean = sum(hist_rates) / len(hist_rates)
                variance = sum((x - mean) ** 2 for x in hist_rates) / len(hist_rates)
                std = math.sqrt(variance)
                if std > 0:
                    z = (current_rate - mean) / std
                    if z < -2.0:
                        anomalies.append(AnomalyItem(
                            anomaly_type="CONVERSION_DROP",
                            severity="WARN",
                            description=f"Conversion rate {current_rate:.1%} is {abs(current_rate - mean):.1%} below {len(hist_rates)}-day avg {mean:.1%}",
                            suggested_action="Check for merchandising or staffing issues; review zone heatmap for dead zones",
                        ))

        # ----------------------------------------------------------------
        # 3. DEAD ZONE — No ZONE_ENTER in last 4 hours
        # ----------------------------------------------------------------
        dead_zone_rows = await conn.fetch(
            """
            SELECT DISTINCT zone_id
            FROM zone_events
            WHERE store_id = $1
              AND event_type = 'zone_entered'
              AND event_time > NOW() - INTERVAL '4 hours'
            """,
            store_id,
        )
        active_zone_ids = {r["zone_id"] for r in dead_zone_rows}

        # Get all zones that have ever had events today
        all_zones_rows = await conn.fetch(
            """
            SELECT DISTINCT zone_id, zone_name
            FROM zone_events
            WHERE store_id = $1
              AND DATE(event_time AT TIME ZONE 'UTC') = CURRENT_DATE
            """,
            store_id,
        )
        for z_row in all_zones_rows:
            if z_row["zone_id"] not in active_zone_ids:
                anomalies.append(AnomalyItem(
                    anomaly_type="DEAD_ZONE",
                    severity="INFO",
                    description=f"Zone {z_row['zone_name']} ({z_row['zone_id']}) had zero entries in last 4 hours",
                    suggested_action=f"Consider promotional activity or signage adjustment in {z_row['zone_name']}",
                ))

    logger.info("anomalies_computed", store_id=store_id, count=len(anomalies))
    return AnomalyResponse(store_id=store_id, anomalies=anomalies)
