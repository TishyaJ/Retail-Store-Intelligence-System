"""
emit.py — Event construction and HTTP batch delivery.

Batch strategy: flush at 50 events OR every 2 seconds, whichever first.
Retry: 3 attempts with 1s/2s/4s exponential backoff.
On final failure: write to buffer.jsonl for manual replay.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

BATCH_MAX_SIZE = 50
FLUSH_INTERVAL_S = 2.0
MAX_RETRIES = 3
BUFFER_PATH = "buffer.jsonl"


class EventEmitter:
    def __init__(self, api_server_url: str, store_id: str, camera_id: str) -> None:
        self.api_url = api_server_url.rstrip("/")
        self.store_id = store_id
        self.camera_id = camera_id
        self._buffer: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()

    def push(self, event: dict[str, Any]) -> None:
        self._buffer.append(event)
        now = time.monotonic()
        if len(self._buffer) >= BATCH_MAX_SIZE or (now - self._last_flush) >= FLUSH_INTERVAL_S:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.monotonic()
        self._post_with_retry(batch)

    def _post_with_retry(self, batch: list[dict]) -> None:
        delays = [1, 2, 4]
        for attempt, delay in enumerate(delays):
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"{self.api_url}/events/ingest",
                        json=batch,
                        headers={"X-Correlation-ID": str(uuid.uuid4())},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    logger.info(
                        "events_posted",
                        store_id=self.store_id,
                        camera_id=self.camera_id,
                        accepted=data.get("accepted"),
                        rejected=data.get("rejected"),
                    )
                    return
            except Exception as e:
                logger.warning(
                    "post_retry",
                    attempt=attempt + 1,
                    error=str(e),
                    camera_id=self.camera_id,
                )
                if attempt < len(delays) - 1:
                    time.sleep(delay)

        # Final failure — write to buffer file
        logger.error("post_failed_writing_buffer", camera_id=self.camera_id, events=len(batch))
        with open(BUFFER_PATH, "a") as f:
            for evt in batch:
                f.write(json.dumps(evt) + "\n")

    def build_entry_exit_event(
        self,
        id_token: str,
        direction: str,   # "entry" | "exit" | "reentry"
        store_code: str,
        store_id: str,
        camera_id: str,
        ts: datetime,
        is_staff: bool = False,
        gender_pred: str | None = None,
        age_pred: int | None = None,
        age_bucket: str | None = None,
        is_face_hidden: bool = False,
        group_id: str | None = None,
        group_size: int | None = None,
        confidence: float = 1.0,
    ) -> dict:
        return {
            "event_type":      direction,
            "id_token":        id_token,
            "store_code":      store_code,
            "store_id":        store_id,
            "camera_id":       camera_id,
            "event_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "is_staff":        is_staff,
            "gender_pred":     gender_pred,
            "age_pred":        age_pred,
            "age_bucket":      age_bucket,
            "is_face_hidden":  is_face_hidden,
            "group_id":        group_id,
            "group_size":      group_size,
            "confidence":      confidence,
        }
