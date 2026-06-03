"""
conftest.py — Shared pytest fixtures for all test modules.

TEST DB STRATEGY:
  - TimescaleDB hypertables and continuous aggregates are wrapped in
    DO $$ BEGIN ... EXCEPTION WHEN others THEN NULL; END $$
    so they degrade gracefully on standard PostgreSQL in test environments.
  - We use pytest-asyncio with asyncio_mode="auto".
  - DB fixtures use standard PostgreSQL (via postgresql fixture from pytest-postgresql
    or the TEST_DATABASE_URL env var for CI environments).
"""
# PROMPT: Generate a comprehensive pytest conftest.py for the Retail Store Intelligence API.
# Key requirements:
#   1. event_loop fixture for asyncio tests
#   2. test_db: async PostgreSQL pool using DATABASE_URL from env (or fallback in-memory mock)
#   3. test_client: FastAPI TestClient with test_db dependency override
#   4. sample_entry_exit_batch: 5 entry + 3 exit events matching sample_events.jsonl schema
#   5. sample_zone_batch: 6 zone events (zone_entered/exited) for ST1076
#   6. sample_queue_batch: 2 queue_completed + 1 queue_abandoned for ST1076
#   7. sample_pos_data: 5 POS rows for ST1008, date 2026-04-10, DD-MM-YYYY format

# CHANGES MADE:
#   - Added import typing and datetime for fixture type hints
#   - Added graceful CREATE EXTENSION IF NOT EXISTS timescaledb (non-fatal)
#   - Added TEST_DATABASE_URL env var override for CI without TimescaleDB

from __future__ import annotations

import asyncio
import os
from datetime import datetime, date
from typing import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# DB pool
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def db_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """
    Async PostgreSQL pool for tests.
    Uses TEST_DATABASE_URL env var (set this in CI).
    Falls back to DATABASE_URL.
    """
    dsn = os.environ.get(
        "TEST_DATABASE_URL",
        os.environ.get("DATABASE_URL", "postgresql://purplle:purplle_secret@localhost:5432/retail_intelligence"),
    ).replace("postgresql+asyncpg://", "postgresql://")

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)

    # Run minimal schema setup (idempotent)
    async with pool.acquire() as conn:
        # TimescaleDB extension (non-fatal if not available)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        except Exception:
            pass

        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entry_exit_events (
                event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                id_token TEXT NOT NULL,
                store_code TEXT,
                store_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_timestamp TIMESTAMPTZ NOT NULL,
                is_staff BOOLEAN NOT NULL DEFAULT false,
                gender_pred TEXT,
                age_pred INT,
                age_bucket TEXT,
                is_face_hidden BOOLEAN DEFAULT false,
                group_id TEXT,
                group_size INT,
                confidence FLOAT NOT NULL DEFAULT 1.0
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS zone_events (
                event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                track_id INT NOT NULL,
                id_token TEXT,
                store_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                zone_name TEXT NOT NULL,
                zone_type TEXT NOT NULL,
                is_revenue_zone BOOLEAN NOT NULL DEFAULT true,
                event_type TEXT NOT NULL,
                event_time TIMESTAMPTZ NOT NULL,
                dwell_ms INT,
                zone_hotspot_x FLOAT,
                zone_hotspot_y FLOAT,
                gender TEXT,
                age INT,
                age_bucket TEXT,
                is_staff BOOLEAN NOT NULL DEFAULT false
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS queue_events (
                queue_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                track_id INT NOT NULL,
                id_token TEXT,
                store_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                zone_name TEXT,
                event_type TEXT NOT NULL,
                queue_join_ts TIMESTAMPTZ NOT NULL,
                queue_served_ts TIMESTAMPTZ,
                queue_exit_ts TIMESTAMPTZ NOT NULL,
                wait_seconds INT NOT NULL,
                queue_position_at_join INT NOT NULL,
                abandoned BOOLEAN NOT NULL DEFAULT false,
                zone_hotspot_x FLOAT,
                zone_hotspot_y FLOAT,
                gender TEXT,
                age INT,
                age_bucket TEXT
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pos_transactions (
                order_id INT PRIMARY KEY,
                order_date DATE NOT NULL,
                order_time TIME NOT NULL,
                store_id TEXT NOT NULL,
                product_id INT NOT NULL,
                brand_name TEXT,
                total_amount NUMERIC(12,2) NOT NULL
            )
        """)

    yield pool

    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM entry_exit_events")
        await conn.execute("DELETE FROM zone_events")
        await conn.execute("DELETE FROM queue_events")
        await conn.execute("DELETE FROM pos_transactions")
    await pool.close()


# ---------------------------------------------------------------------------
# FastAPI Test Client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client(db_pool):
    """FastAPI TestClient with DB pool dependency overridden."""
    from app.main import app
    from app.database import get_pool

    app.dependency_overrides[get_pool] = lambda: db_pool
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Event batch fixtures — match actual sample_events.jsonl schema exactly
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entry_exit_batch() -> list[dict]:
    """5 entry + 3 exit events matching sample_events.jsonl schema."""
    return [
        {
            "id_token": f"ID_6000{i}",
            "store_code": "store_1076",
            "camera_id": "cam1",
            "event_type": "entry",
            "event_timestamp": f"2026-03-08T10:0{i}:00.000000",
            "is_staff": False,
            "gender_pred": "F" if i % 2 == 0 else "M",
            "age_pred": 25 + i * 2,
            "age_bucket": "25-34",
            "is_face_hidden": False,
            "group_id": None,
            "group_size": None,
            "confidence": 0.91,
        }
        for i in range(1, 6)
    ] + [
        {
            "id_token": f"ID_6000{i}",
            "store_code": "store_1076",
            "camera_id": "cam1",
            "event_type": "exit",
            "event_timestamp": f"2026-03-08T11:0{i}:00.000000",
            "is_staff": False,
            "gender_pred": "F" if i % 2 == 0 else "M",
            "age_pred": 25 + i * 2,
            "age_bucket": "25-34",
            "is_face_hidden": False,
            "group_id": None,
            "group_size": None,
            "confidence": 0.88,
        }
        for i in range(1, 4)
    ]


@pytest.fixture
def sample_zone_batch() -> list[dict]:
    """6 zone events for ST1076 PURPLLE_MUM_1076_Z01, Z02, Z03."""
    events = []
    zones = [
        ("PURPLLE_MUM_1076_Z01", "Left Shelf"),
        ("PURPLLE_MUM_1076_Z02", "Center Display"),
        ("PURPLLE_MUM_1076_Z03", "Lipstick Aisle"),
    ]
    for i, (zone_id, zone_name) in enumerate(zones):
        events.append({
            "track_id": 100 + i,
            "id_token": f"ID_6000{i+1}",
            "store_id": "ST1076",
            "camera_id": "CAM2",
            "zone_id": zone_id,
            "zone_name": zone_name,
            "zone_type": "SHELF" if "Shelf" in zone_name or "Aisle" in zone_name else "DISPLAY",
            "is_revenue_zone": "Yes",
            "event_type": "zone_entered",
            "event_time": f"2026-03-08T10:1{i}:00.000000",
            "dwell_ms": None,
            "zone_hotspot_x": 412.6 + i * 100,
            "zone_hotspot_y": 238.4 + i * 50,
            "gender": "F",
            "age": 28,
            "age_bucket": "25-34",
            "is_staff": False,
        })
        events.append({
            "track_id": 100 + i,
            "id_token": f"ID_6000{i+1}",
            "store_id": "ST1076",
            "camera_id": "CAM2",
            "zone_id": zone_id,
            "zone_name": zone_name,
            "zone_type": "SHELF" if "Shelf" in zone_name or "Aisle" in zone_name else "DISPLAY",
            "is_revenue_zone": "Yes",
            "event_type": "zone_exited",
            "event_time": f"2026-03-08T10:1{i}:30.000000",
            "dwell_ms": 30000 + i * 5000,
            "zone_hotspot_x": 412.6 + i * 100,
            "zone_hotspot_y": 238.4 + i * 50,
            "gender": "F",
            "age": 28,
            "age_bucket": "25-34",
            "is_staff": False,
        })
    return events


@pytest.fixture
def sample_queue_batch() -> list[dict]:
    """2 queue_completed + 1 queue_abandoned for ST1076 billing zone."""
    import uuid
    return [
        {
            "queue_event_id": str(uuid.uuid4()),
            "event_type": "queue_completed",
            "track_id": 200,
            "id_token": "ID_60001",
            "store_id": "ST1076",
            "camera_id": "CAM5",
            "zone_id": "PURPLLE_MUM_1076_Z_BILLING_01",
            "zone_name": "Billing Counter Queue",
            "queue_join_ts": "2026-03-08T11:00:00.000000",
            "queue_served_ts": "2026-03-08T11:02:30.000000",
            "queue_exit_ts": "2026-03-08T11:03:00.000000",
            "wait_seconds": 180,
            "queue_position_at_join": 2,
            "abandoned": False,
        },
        {
            "queue_event_id": str(uuid.uuid4()),
            "event_type": "queue_completed",
            "track_id": 201,
            "id_token": "ID_60002",
            "store_id": "ST1076",
            "camera_id": "CAM5",
            "zone_id": "PURPLLE_MUM_1076_Z_BILLING_01",
            "zone_name": "Billing Counter Queue",
            "queue_join_ts": "2026-03-08T11:05:00.000000",
            "queue_served_ts": "2026-03-08T11:07:00.000000",
            "queue_exit_ts": "2026-03-08T11:07:30.000000",
            "wait_seconds": 150,
            "queue_position_at_join": 1,
            "abandoned": False,
        },
        {
            "queue_event_id": str(uuid.uuid4()),
            "event_type": "queue_abandoned",
            "track_id": 202,
            "id_token": "ID_60003",
            "store_id": "ST1076",
            "camera_id": "CAM5",
            "zone_id": "PURPLLE_MUM_1076_Z_BILLING_01",
            "zone_name": "Billing Counter Queue",
            "queue_join_ts": "2026-03-08T11:10:00.000000",
            "queue_served_ts": None,
            "queue_exit_ts": "2026-03-08T11:15:00.000000",
            "wait_seconds": 300,
            "queue_position_at_join": 4,
            "abandoned": True,
        },
    ]


@pytest.fixture
def sample_pos_data() -> list[dict]:
    """5 POS rows for ST1008, date 10-04-2026 (DD-MM-YYYY format in source CSV)."""
    return [
        {"order_id": 1, "order_date": "10-04-2026", "order_time": "10:15:00", "store_id": "ST1008",
         "product_id": 101, "brand_name": "Lakme", "total_amount": 450.00},
        {"order_id": 2, "order_date": "10-04-2026", "order_time": "10:15:00", "store_id": "ST1008",
         "product_id": 102, "brand_name": "Maybelline", "total_amount": 320.00},
        {"order_id": 3, "order_date": "10-04-2026", "order_time": "11:30:00", "store_id": "ST1008",
         "product_id": 201, "brand_name": "L'Oreal", "total_amount": 870.00},
        {"order_id": 4, "order_date": "10-04-2026", "order_time": "12:45:00", "store_id": "ST1008",
         "product_id": 301, "brand_name": "NYX", "total_amount": 560.00},
        {"order_id": 5, "order_date": "10-04-2026", "order_time": "14:00:00", "store_id": "ST1008",
         "product_id": 401, "brand_name": "MAC", "total_amount": 1200.00},
    ]
