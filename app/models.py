"""
Pydantic v2 schemas for all three event types from sample_events.jsonl.

IMPORTANT: The actual wire format uses 3 SEPARATE schemas (not a unified schema).
See .agents/specs/retail-store-intelligence/design.md §3.1 for routing logic.
"""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_store_code(raw: str) -> str:
    """Convert 'store_1076' → 'ST1076'. Handles already-normalised IDs."""
    if raw and raw.startswith("store_"):
        return "ST" + raw[6:]
    return raw


# ---------------------------------------------------------------------------
# Entry / Exit Event
# From sample_events.jsonl lines 1–4
# ---------------------------------------------------------------------------

class EntryExitEventType(str, Enum):
    entry   = "entry"
    exit    = "exit"
    reentry = "reentry"


class EntryExitEvent(BaseModel):
    id_token:         str
    store_code:       Optional[str] = None      # raw: "store_1076"
    store_id:         Optional[str] = None      # normalised: "ST1076"
    camera_id:        str
    event_type:       EntryExitEventType
    event_timestamp:  datetime
    is_staff:         bool = False
    gender_pred:      Optional[str] = None
    age_pred:         Optional[int] = None
    age_bucket:       Optional[str] = None
    is_face_hidden:   bool = False
    group_id:         Optional[str] = None
    group_size:       Optional[int] = None
    confidence:       float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def resolve_store_id(self) -> "EntryExitEvent":
        if self.store_id is None and self.store_code:
            self.store_id = normalise_store_code(self.store_code)
        if self.store_id is None:
            raise ValueError("Either store_id or store_code must be provided")
        return self

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Zone Event
# From sample_events.jsonl lines 5–10
# ---------------------------------------------------------------------------

class ZoneEventType(str, Enum):
    zone_entered = "zone_entered"
    zone_exited  = "zone_exited"
    zone_dwell   = "zone_dwell"


class ZoneType(str, Enum):
    SHELF   = "SHELF"
    DISPLAY = "DISPLAY"
    BILLING = "BILLING"


class ZoneEvent(BaseModel):
    track_id:         int
    id_token:         Optional[str] = None       # linked to entry_exit_events
    store_id:         str
    camera_id:        str
    zone_id:          str
    zone_name:        str
    zone_type:        str                         # SHELF / DISPLAY / BILLING
    is_revenue_zone:  Any = True                  # may be "Yes"/"No" string or bool
    event_type:       ZoneEventType
    event_time:       datetime
    dwell_ms:         Optional[int] = None        # null for zone_entered
    zone_hotspot_x:   Optional[float] = None
    zone_hotspot_y:   Optional[float] = None
    gender:           Optional[str] = None
    age:              Optional[int] = None
    age_bucket:       Optional[str] = None
    is_staff:         bool = False

    @field_validator("is_revenue_zone", mode="before")
    @classmethod
    def coerce_revenue_zone(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("yes", "true", "1")
        return bool(v)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Queue Event
# From sample_events.jsonl lines 11–13
# ---------------------------------------------------------------------------

class QueueEventType(str, Enum):
    queue_completed = "queue_completed"
    queue_abandoned = "queue_abandoned"


class QueueEvent(BaseModel):
    queue_event_id:         UUID = Field(default_factory=uuid4)
    track_id:               int
    id_token:               Optional[str] = None
    store_id:               str
    camera_id:              str
    zone_id:                str
    zone_name:              Optional[str] = None
    event_type:             QueueEventType
    queue_join_ts:          datetime
    queue_served_ts:        Optional[datetime] = None    # null if abandoned
    queue_exit_ts:          datetime
    wait_seconds:           int
    queue_position_at_join: int
    abandoned:              bool = False
    zone_hotspot_x:         Optional[float] = None
    zone_hotspot_y:         Optional[float] = None
    gender:                 Optional[str] = None
    age:                    Optional[int] = None
    age_bucket:             Optional[str] = None

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Ingest Request/Response
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    errors:   list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# API response shapes
# ---------------------------------------------------------------------------

class StoreMetricsResponse(BaseModel):
    store_id:           str
    date:               str
    unique_visitors:    int
    conversion_rate:    float
    avg_dwell_per_zone: dict[str, Optional[float]]
    queue_depth:        int
    abandonment_rate:   float


class FunnelStage(BaseModel):
    stage:        str
    count:        int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    date:     str
    stages:   list[FunnelStage]


class HeatmapZone(BaseModel):
    zone_id:          str
    zone_name:        str
    visit_frequency:  int       # normalised 0–100
    avg_dwell_ms:     Optional[float]
    data_confidence:  bool      # True if ≥20 unique sessions


class HeatmapResponse(BaseModel):
    store_id: str
    zones:    list[HeatmapZone]


class AnomalyItem(BaseModel):
    anomaly_type:     str
    severity:         str       # INFO / WARN / CRITICAL
    description:      str
    suggested_action: str


class AnomalyResponse(BaseModel):
    store_id:  str
    anomalies: list[AnomalyItem]


class HealthResponse(BaseModel):
    status:                        str
    last_event_timestamp_per_store: dict[str, Optional[str]]
    warnings:                      list[str]
