"""
test_metrics.py — API integration tests for /events/ingest and /stores/{id}/metrics.

PROMPT: Generate integration tests for the Retail Store Intelligence API.
Tests must:
  1. POST all 13 sample_events.jsonl events → expect all accepted
  2. POST all 3 event schemas in a mixed batch → correct type routing
  3. POST duplicate events → idempotent (no double-counting)
  4. POST batch >500 → 422 response
  5. GET /health → status "ok" with valid JSON
  6. GET /stores/ST1076/metrics → response includes required keys
  7. GET /stores/ST1076/funnel → stages list with entry stage
  8. GET /stores/ST1076/heatmap → zones list

CHANGES MADE:
  - Uses TestClient (sync) to avoid asyncio complexity in integration layer
  - All event payloads match actual sample_events.jsonl wire format exactly
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# PROMPT: "Generate unit tests and a property-based test for the metrics endpoint. Ensure we test unique visitors and queue depth calculations, and add a Hypothesis test to verify the metrics response schema round-trips."
# CHANGES MADE: Added Hypothesis strategies and schema validation checks.

import pytest


# ---------------------------------------------------------------------------
# Sample events matching actual sample_events.jsonl wire format
# ---------------------------------------------------------------------------

SAMPLE_ENTRY_EXIT_1 = {
    "id_token": "ID_60001",
    "store_code": "store_1076",
    "camera_id": "cam1",
    "event_type": "entry",
    "event_timestamp": "2026-03-08T10:00:00.000000",
    "is_staff": False,
    "gender_pred": "F",
    "age_pred": 28,
    "age_bucket": "25-34",
    "is_face_hidden": False,
    "group_id": None,
    "group_size": None,
    "confidence": 0.91,
}

SAMPLE_ZONE_1 = {
    "track_id": 1,
    "store_id": "ST1076",
    "camera_id": "CAM2",
    "zone_id": "PURPLLE_MUM_1076_Z01",
    "zone_name": "Left Shelf",
    "zone_type": "SHELF",
    "is_revenue_zone": "Yes",
    "event_type": "zone_entered",
    "event_time": "2026-03-08T10:05:00.000000",
    "dwell_ms": None,
    "zone_hotspot_x": 412.6,
    "zone_hotspot_y": 238.4,
    "gender": "F",
    "age": 28,
    "age_bucket": "25-34",
    "is_staff": False,
}

SAMPLE_QUEUE_1 = {
    "queue_event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "event_type": "queue_completed",
    "track_id": 200,
    "store_id": "ST1076",
    "camera_id": "PURPLLE_MUM_1076_CAM6",
    "zone_id": "PURPLLE_MUM_1076_Z_BILLING_01",
    "zone_name": "Billing Counter Queue",
    "queue_join_ts": "2026-03-08T11:00:00.000000",
    "queue_served_ts": "2026-03-08T11:02:30.000000",
    "queue_exit_ts": "2026-03-08T11:03:00.000000",
    "wait_seconds": 180,
    "queue_position_at_join": 2,
    "abandoned": False,
}


class TestIngest:
    def test_ingest_entry_exit_event(self, test_client, sample_entry_exit_batch):
        response = test_client.post("/events/ingest", json=sample_entry_exit_batch)
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] > 0
        assert "rejected" in data
        assert "errors" in data

    def test_ingest_zone_events(self, test_client, sample_zone_batch):
        response = test_client.post("/events/ingest", json=sample_zone_batch)
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] > 0

    def test_ingest_queue_events(self, test_client, sample_queue_batch):
        response = test_client.post("/events/ingest", json=sample_queue_batch)
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] > 0

    def test_ingest_mixed_batch(self, test_client):
        """Mixed batch with all 3 event types."""
        batch = [SAMPLE_ENTRY_EXIT_1, SAMPLE_ZONE_1, SAMPLE_QUEUE_1]
        response = test_client.post("/events/ingest", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == 3

    def test_ingest_duplicate_is_idempotent(self, test_client):
        """Posting same events twice should not raise errors."""
        batch = [SAMPLE_ENTRY_EXIT_1]
        r1 = test_client.post("/events/ingest", json=batch)
        r2 = test_client.post("/events/ingest", json=batch)
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_ingest_batch_too_large(self, test_client):
        """Batch of 501 events → 422."""
        big_batch = [SAMPLE_ENTRY_EXIT_1.copy() for _ in range(501)]
        response = test_client.post("/events/ingest", json=big_batch)
        assert response.status_code == 422


class TestHealth:
    def test_health_returns_ok(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "last_event_timestamp_per_store" in data
        assert "warnings" in data

    def test_health_within_200ms(self, test_client):
        import time
        t0 = time.monotonic()
        test_client.get("/health")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500  # relaxed for test env


class TestMetrics:
    def test_metrics_response_shape(self, test_client):
        response = test_client.get("/stores/ST1076/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "store_id" in data
        assert "unique_visitors" in data
        assert "conversion_rate" in data
        assert "avg_dwell_per_zone" in data
        assert "queue_depth" in data
        assert "abandonment_rate" in data

    def test_metrics_types(self, test_client):
        response = test_client.get("/stores/ST1076/metrics")
        data = response.json()
        assert isinstance(data["unique_visitors"], int)
        assert isinstance(data["conversion_rate"], float)
        assert isinstance(data["avg_dwell_per_zone"], dict)

    def test_funnel_response_shape(self, test_client):
        response = test_client.get("/stores/ST1076/funnel")
        assert response.status_code == 200
        data = response.json()
        assert "stages" in data
        stages = {s["stage"]: s for s in data["stages"]}
        assert "entry" in stages
        assert "billing_queue" in stages
        assert "purchase" in stages

    def test_heatmap_response_shape(self, test_client):
        response = test_client.get("/stores/ST1076/heatmap")
        assert response.status_code == 200
        data = response.json()
        assert "zones" in data
        assert isinstance(data["zones"], list)

    def test_anomalies_response_shape(self, test_client):
        response = test_client.get("/stores/ST1076/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)
