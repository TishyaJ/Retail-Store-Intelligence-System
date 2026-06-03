"""
POST /events/ingest — Batch event ingestion with idempotency.

Accepts up to 500 events in a mixed batch (entry/exit, zone, queue).
Event type is auto-detected from payload field fingerprints.

Idempotency strategy: All 3 event tables use asyncpg.UniqueViolationError
catch (try/except) instead of ON CONFLICT DO NOTHING.
Reason: TimescaleDB hypertables require composite PKs (event_id + partition_time)
on all 3 event tables, which prevents ON CONFLICT targeting a single column.
The _insert() helper transparently handles duplicate detection for all tables.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from .database import get_pool
from .models import (
    EntryExitEvent,
    IngestResponse,
    QueueEvent,
    ZoneEvent,
)

logger = structlog.get_logger()
router = APIRouter()

MAX_BATCH = 500


def detect_event_type(raw: dict[str, Any]) -> str:
    """Fingerprint the event type from payload fields."""
    if "queue_event_id" in raw or "queue_join_ts" in raw:
        return "queue"
    if "event_time" in raw and "zone_id" in raw and "track_id" in raw:
        return "zone"
    if "event_timestamp" in raw or "store_code" in raw:
        return "entry_exit"
    et = raw.get("event_type", "")
    if et in ("zone_entered", "zone_exited", "zone_dwell"):
        return "zone"
    if et in ("queue_completed", "queue_abandoned"):
        return "queue"
    return "entry_exit"


async def _insert(conn, query: str, *args) -> bool:
    """
    Execute INSERT, returning True if inserted, False if duplicate.
    Catches asyncpg.UniqueViolationError for idempotent behaviour.
    """
    try:
        result = await conn.execute(query, *args)
        return result != "INSERT 0 0"
    except asyncpg.UniqueViolationError:
        return False


@router.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(
    batch: list[dict[str, Any]],
    pool: asyncpg.Pool = Depends(get_pool),
) -> IngestResponse:
    t0 = time.monotonic()

    if len(batch) > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(batch)} exceeds maximum of {MAX_BATCH}",
        )

    entry_exit_rows: list[dict] = []
    zone_rows: list[dict] = []
    queue_rows: list[dict] = []
    errors: list[dict] = []

    for idx, raw in enumerate(batch):
        try:
            etype = detect_event_type(raw)
            if etype == "entry_exit":
                evt = EntryExitEvent(**raw)
                entry_exit_rows.append(evt.model_dump())
            elif etype == "zone":
                evt = ZoneEvent(**raw)
                zone_rows.append(evt.model_dump())
            else:
                evt = QueueEvent(**raw)
                queue_rows.append(evt.model_dump())
        except (ValidationError, Exception) as e:
            errors.append({"index": idx, "detail": str(e)})

    accepted = 0

    async with pool.acquire() as conn:
        # --- entry_exit_events (TimescaleDB composite PK — use try/except) ---
        for row in entry_exit_rows:
            try:
                inserted = await _insert(
                    conn,
                    """
                    INSERT INTO entry_exit_events
                      (event_id, id_token, store_code, store_id, camera_id,
                       event_type, event_timestamp, is_staff,
                       gender_pred, age_pred, age_bucket,
                       is_face_hidden, group_id, group_size, confidence)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                    """,
                    row.get("event_id") or uuid.uuid4(),
                    row["id_token"],
                    row.get("store_code"),
                    row["store_id"],
                    row["camera_id"],
                    row["event_type"],
                    row["event_timestamp"],
                    row["is_staff"],
                    row.get("gender_pred"),
                    row.get("age_pred"),
                    row.get("age_bucket"),
                    row.get("is_face_hidden", False),
                    row.get("group_id"),
                    row.get("group_size"),
                    row.get("confidence", 1.0),
                )
                if inserted:
                    accepted += 1
            except Exception as e:
                errors.append({"detail": str(e), "row": row.get("id_token")})

        # --- zone_events (TimescaleDB composite PK — use try/except) ---
        for row in zone_rows:
            try:
                inserted = await _insert(
                    conn,
                    """
                    INSERT INTO zone_events
                      (event_id, track_id, id_token, store_id, camera_id,
                       zone_id, zone_name, zone_type, is_revenue_zone,
                       event_type, event_time, dwell_ms,
                       zone_hotspot_x, zone_hotspot_y,
                       gender, age, age_bucket, is_staff)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
                    """,
                    row.get("event_id") or uuid.uuid4(),
                    row["track_id"],
                    row.get("id_token"),
                    row["store_id"],
                    row["camera_id"],
                    row["zone_id"],
                    row["zone_name"],
                    row["zone_type"],
                    bool(row.get("is_revenue_zone", True)),
                    row["event_type"],
                    row["event_time"],
                    row.get("dwell_ms"),
                    row.get("zone_hotspot_x"),
                    row.get("zone_hotspot_y"),
                    row.get("gender"),
                    row.get("age"),
                    row.get("age_bucket"),
                    row.get("is_staff", False),
                )
                if inserted:
                    accepted += 1
            except Exception as e:
                errors.append({"detail": str(e), "row": row.get("zone_id")})

        # --- queue_events (TimescaleDB composite PK — use try/except) ---
        for row in queue_rows:
            try:
                inserted = await _insert(
                    conn,
                    """
                    INSERT INTO queue_events
                      (queue_event_id, track_id, id_token, store_id, camera_id,
                       zone_id, zone_name, event_type,
                       queue_join_ts, queue_served_ts, queue_exit_ts,
                       wait_seconds, queue_position_at_join, abandoned,
                       zone_hotspot_x, zone_hotspot_y, gender, age, age_bucket)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
                    """,
                    row.get("queue_event_id") or uuid.uuid4(),
                    row["track_id"],
                    row.get("id_token"),
                    row["store_id"],
                    row["camera_id"],
                    row["zone_id"],
                    row.get("zone_name"),
                    row["event_type"],
                    row["queue_join_ts"],
                    row.get("queue_served_ts"),
                    row["queue_exit_ts"],
                    row["wait_seconds"],
                    row["queue_position_at_join"],
                    row.get("abandoned", False),
                    row.get("zone_hotspot_x"),
                    row.get("zone_hotspot_y"),
                    row.get("gender"),
                    row.get("age"),
                    row.get("age_bucket"),
                )
                if inserted:
                    accepted += 1
            except Exception as e:
                errors.append({"detail": str(e), "row": str(row.get("queue_event_id"))})

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    rejected = len(errors)

    if rejected > 0:
        logger.warning(
            "ingest_batch_partial",
            accepted=accepted,
            rejected=rejected,
            first_error=errors[0].get("detail"),
            latency_ms=latency_ms,
        )
    else:
        logger.info("ingest_batch_complete", accepted=accepted, latency_ms=latency_ms)

    return IngestResponse(accepted=accepted, rejected=rejected, errors=errors)
