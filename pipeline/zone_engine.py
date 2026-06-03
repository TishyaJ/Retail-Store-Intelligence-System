"""
zone_engine.py — Spatial hysteresis state machines for zone event emission.

Per-(id_token, zone_id) state machine:
  OUTSIDE → INSIDE on foot-point entering inner_polygon
  INSIDE  → OUTSIDE on foot-point exiting outer_polygon
  (between polygons: no state change — debounces tracking jitter)

ZONE_DWELL emitted every 30s while INSIDE.
Billing queue logic: tracks queue_position, sets abandoned flag.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from shapely.geometry import Point, Polygon

logger = structlog.get_logger()

DWELL_INTERVAL_SECONDS = 30


@dataclass
class ZoneDef:
    zone_id:        str
    zone_name:      str
    zone_type:      str
    is_revenue_zone: bool
    store_id:       str
    inner_poly:     Polygon
    outer_poly:     Polygon


@dataclass
class VisitorZoneState:
    state: str = "OUTSIDE"            # OUTSIDE | INSIDE
    enter_ts: Optional[float] = None  # epoch seconds
    last_dwell_ts: Optional[float] = None


class ZoneEngine:
    def __init__(
        self,
        zones_config_path: str = "store_layout/zones.json",
        store_id: str = "ST1076",
    ) -> None:
        self.store_id = store_id
        self.zones: dict[str, ZoneDef] = {}
        self._load_zones(zones_config_path, store_id)

        # State: {(id_token, zone_id): VisitorZoneState}
        self._states: dict[tuple[str, str], VisitorZoneState] = {}

        # Billing queue tracking: {id_token: {join_ts, position}}
        self._billing_queue: dict[str, dict] = {}
        self._billing_count: int = 0

    def _load_zones(self, config_path: str, store_id: str) -> None:
        try:
            with open(config_path) as f:
                config = json.load(f)
            store_conf = config.get("stores", {}).get(store_id, {})
            for z in store_conf.get("zones", []):
                inner = Polygon(z["inner_polygon"])
                outer = Polygon(z["outer_polygon"])
                self.zones[z["zone_id"]] = ZoneDef(
                    zone_id=z["zone_id"],
                    zone_name=z["zone_name"],
                    zone_type=z["zone_type"],
                    is_revenue_zone=z.get("is_revenue_zone", True),
                    store_id=store_id,
                    inner_poly=inner,
                    outer_poly=outer,
                )
            logger.info("zones_loaded", store_id=store_id, count=len(self.zones))
        except Exception as e:
            logger.error("zone_config_load_failed", error=str(e))

    def process_track(
        self,
        id_token: str,
        camera_id: str,
        ground_xy: tuple[float, float],
        is_staff: bool,
        ts: datetime,
        track_id: int,
        gender: Optional[str] = None,
        age: Optional[int] = None,
        age_bucket: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Process one track's ground-plane position against all zones.
        Returns list of events to emit.
        """
        events = []
        pt = Point(ground_xy)
        now_epoch = ts.timestamp()

        for zone_id, zone in self.zones.items():
            state_key = (id_token, zone_id)
            state = self._states.get(state_key, VisitorZoneState())

            in_inner = zone.inner_poly.contains(pt)
            in_outer = zone.outer_poly.contains(pt)

            if state.state == "OUTSIDE" and in_inner:
                # OUTSIDE → INSIDE transition
                state.state = "INSIDE"
                state.enter_ts = now_epoch
                state.last_dwell_ts = now_epoch
                self._states[state_key] = state

                evt = self._build_zone_event(
                    event_type="zone_entered",
                    zone=zone, id_token=id_token,
                    camera_id=camera_id, track_id=track_id,
                    ts=ts, dwell_ms=None, ground_xy=ground_xy,
                    is_staff=is_staff, gender=gender, age=age, age_bucket=age_bucket,
                )
                events.append(evt)

                # Billing queue logic
                if zone.zone_type == "BILLING":
                    self._billing_count += 1
                    self._billing_queue[id_token] = {
                        "join_ts": ts,
                        "position": self._billing_count,
                        "zone_id": zone_id,
                        "zone_name": zone.zone_name,
                    }

            elif state.state == "INSIDE":
                if not in_outer:
                    # INSIDE → OUTSIDE transition (exited outer polygon)
                    dwell_ms = int((now_epoch - state.enter_ts) * 1000) if state.enter_ts else 0
                    state.state = "OUTSIDE"
                    state.enter_ts = None
                    state.last_dwell_ts = None
                    self._states[state_key] = state

                    evt = self._build_zone_event(
                        event_type="zone_exited",
                        zone=zone, id_token=id_token,
                        camera_id=camera_id, track_id=track_id,
                        ts=ts, dwell_ms=dwell_ms, ground_xy=ground_xy,
                        is_staff=is_staff, gender=gender, age=age, age_bucket=age_bucket,
                    )
                    events.append(evt)

                    # Billing: emit queue event
                    if zone.zone_type == "BILLING" and id_token in self._billing_queue:
                        queue_info = self._billing_queue.pop(id_token)
                        self._billing_count = max(0, self._billing_count - 1)
                        queue_evt = self._build_queue_event(
                            id_token=id_token, track_id=track_id,
                            camera_id=camera_id, zone=zone,
                            queue_info=queue_info, exit_ts=ts,
                            abandoned=True,  # Will be corrected when POS match found
                        )
                        events.append(queue_evt)

                else:
                    # Still INSIDE — check DWELL timer
                    if state.last_dwell_ts and (now_epoch - state.last_dwell_ts) >= DWELL_INTERVAL_SECONDS:
                        dwell_ms = int((now_epoch - state.enter_ts) * 1000) if state.enter_ts else 0
                        state.last_dwell_ts = now_epoch
                        self._states[state_key] = state
                        evt = self._build_zone_event(
                            event_type="zone_dwell",
                            zone=zone, id_token=id_token,
                            camera_id=camera_id, track_id=track_id,
                            ts=ts, dwell_ms=dwell_ms, ground_xy=ground_xy,
                            is_staff=is_staff, gender=gender, age=age, age_bucket=age_bucket,
                        )
                        events.append(evt)

        return events

    def synthesise_exit_events(
        self, id_token: str, camera_id: str, exit_ts: datetime, track_id: int
    ) -> list[dict]:
        """Emit synthetic zone_exited for all open zone states at session close."""
        events = []
        now_epoch = exit_ts.timestamp()
        for (tok, zone_id), state in list(self._states.items()):
            if tok != id_token or state.state != "INSIDE":
                continue
            zone = self.zones.get(zone_id)
            if not zone:
                continue
            dwell_ms = int((now_epoch - state.enter_ts) * 1000) if state.enter_ts else 0
            events.append(self._build_zone_event(
                event_type="zone_exited",
                zone=zone, id_token=id_token,
                camera_id=camera_id, track_id=track_id,
                ts=exit_ts, dwell_ms=dwell_ms, ground_xy=(0.0, 0.0),
                is_staff=False, gender=None, age=None, age_bucket=None,
            ))
            state.state = "OUTSIDE"
        return events

    @staticmethod
    def _build_zone_event(
        event_type: str, zone: ZoneDef, id_token: str,
        camera_id: str, track_id: int, ts: datetime,
        dwell_ms: Optional[int], ground_xy: tuple[float, float],
        is_staff: bool, gender: Optional[str], age: Optional[int], age_bucket: Optional[str],
    ) -> dict:
        return {
            "event_type":     event_type,
            "track_id":       track_id,
            "id_token":       id_token,
            "store_id":       zone.store_id,
            "camera_id":      camera_id,
            "zone_id":        zone.zone_id,
            "zone_name":      zone.zone_name,
            "zone_type":      zone.zone_type,
            "is_revenue_zone": "Yes" if zone.is_revenue_zone else "No",
            "event_time":     ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "dwell_ms":       dwell_ms,
            "zone_hotspot_x": round(ground_xy[0], 1),
            "zone_hotspot_y": round(ground_xy[1], 1),
            "gender":         gender,
            "age":            age,
            "age_bucket":     age_bucket,
            "is_staff":       is_staff,
        }

    @staticmethod
    def _build_queue_event(
        id_token: str, track_id: int, camera_id: str,
        zone: ZoneDef, queue_info: dict, exit_ts: datetime, abandoned: bool,
    ) -> dict:
        join_ts: datetime = queue_info["join_ts"]
        wait_s = int((exit_ts.timestamp() - join_ts.timestamp()))
        return {
            "queue_event_id":        str(uuid.uuid4()),
            "event_type":            "queue_abandoned" if abandoned else "queue_completed",
            "track_id":              track_id,
            "id_token":              id_token,
            "store_id":              zone.store_id,
            "camera_id":             camera_id,
            "zone_id":               zone.zone_id,
            "zone_name":             zone.zone_name,
            "zone_type":             zone.zone_type,
            "is_revenue_zone":       "Yes" if zone.is_revenue_zone else "No",
            "queue_join_ts":         join_ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "queue_served_ts":       None,
            "queue_exit_ts":         exit_ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "wait_seconds":          wait_s,
            "queue_position_at_join": queue_info["position"],
            "abandoned":             abandoned,
            "zone_hotspot_x":        None,
            "zone_hotspot_y":        None,
        }
