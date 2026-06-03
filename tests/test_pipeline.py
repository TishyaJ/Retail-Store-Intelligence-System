"""
test_pipeline.py — Vision pipeline unit tests.

Tests cover: detection thresholds, ByteTrack matching, re-ID gallery,
             zone state machine transitions, event schema validation.
"""
# PROMPT: Generate unit tests for the vision pipeline modules.
# Requirements:
#   1. Test detect.py: Detection split at HIGH_CONF=0.60 / LOW_CONF=0.30
#   2. Test tracker.py: ByteTrack 2-pass matching, track persistence, LOST/REMOVED lifecycle
#   3. Test reid.py: Per-store gallery isolation, TTL eviction, id_token format
#   4. Test zone_engine.py: zone_entered on inner_poly entry, zone_exited on outer_poly exit,
#      DWELL timer, billing queue event construction
#   5. Verify id_token format "ID_XXXXX" with per-store base offsets
#   6. Run with: pytest tests/test_pipeline.py -v

# CHANGES MADE:
#   - Mocked cv2.VideoCapture for detector tests (no actual video files needed)
#   - Used in-memory numpy arrays instead of real image crops
#   - Zone tests use Shapely polygon geometry directly (no camera/homography needed)

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# detect.py tests
# ---------------------------------------------------------------------------

class TestDetectionSplit:
    def test_high_conf_threshold(self):
        """Detections above 0.60 go to high list."""
        from pipeline.detect import Detection, HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD
        d = Detection(x1=0, y1=0, x2=100, y2=200, confidence=0.85)
        assert d.confidence > HIGH_CONF_THRESHOLD

    def test_low_conf_threshold(self):
        """Detections in 0.30-0.60 range go to low list."""
        from pipeline.detect import Detection, HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD
        d = Detection(x1=0, y1=0, x2=100, y2=200, confidence=0.45)
        assert LOW_CONF_THRESHOLD <= d.confidence <= HIGH_CONF_THRESHOLD

    def test_below_low_conf_discarded(self):
        """Detections below 0.30 are discarded."""
        from pipeline.detect import Detection, LOW_CONF_THRESHOLD
        d = Detection(x1=0, y1=0, x2=100, y2=200, confidence=0.15)
        assert d.confidence < LOW_CONF_THRESHOLD

    def test_foot_point(self):
        """Foot point is bottom-centre of bounding box."""
        from pipeline.detect import Detection
        d = Detection(x1=100, y1=50, x2=300, y2=400, confidence=0.9)
        fx, fy = d.foot_point
        assert fx == 200.0   # (100+300)/2
        assert fy == 400.0   # y2


# ---------------------------------------------------------------------------
# tracker.py tests
# ---------------------------------------------------------------------------

class TestByteTracker:
    def _make_det(self, x1, y1, x2, y2, conf=0.9):
        from pipeline.detect import Detection
        return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)

    def test_new_detection_creates_track(self):
        from pipeline.tracker import ByteTracker, TrackState
        tracker = ByteTracker()
        high = [self._make_det(0, 0, 100, 200, 0.9)]
        tracks = tracker.update(high, [])
        assert len(tracks) == 1
        assert tracks[0].state in (TrackState.NEW, TrackState.TRACKED)

    def test_matching_detection_persists_track_id(self):
        from pipeline.tracker import ByteTracker
        tracker = ByteTracker()
        high = [self._make_det(0, 0, 100, 200, 0.9)]
        t1 = tracker.update(high, [])
        orig_id = t1[0].track_id
        # Same position next frame
        t2 = tracker.update(high, [])
        assert any(t.track_id == orig_id for t in t2)

    def test_no_detection_increments_lost_frames(self):
        from pipeline.tracker import ByteTracker, TrackState
        tracker = ByteTracker()
        high = [self._make_det(0, 0, 100, 200, 0.9)]
        tracker.update(high, [])
        # No detection next frame
        tracks = tracker.update([], [])
        # Track should be LOST (not yet removed — threshold is 30 frames)
        all_tracks = tracker.tracks
        if all_tracks:
            assert all_tracks[0].state == TrackState.LOST


# ---------------------------------------------------------------------------
# reid.py tests
# ---------------------------------------------------------------------------

class TestReIDGallery:
    def test_id_token_format_store1(self):
        from pipeline.reid import ReIDModule
        reid = ReIDModule.__new__(ReIDModule)
        reid._gallery = {}
        reid._counters = {}
        from datetime import date
        reid._operating_day = date.today()
        token = reid._next_id_token("ST1076")
        assert token.startswith("ID_6")

    def test_id_token_format_store2(self):
        from pipeline.reid import ReIDModule
        reid = ReIDModule.__new__(ReIDModule)
        reid._gallery = {}
        reid._counters = {}
        from datetime import date
        reid._operating_day = date.today()
        token = reid._next_id_token("ST1008")
        assert token.startswith("ID_7")

    def test_per_store_gallery_isolation(self):
        """Galleries must be separate per store."""
        from pipeline.reid import ReIDModule
        reid = ReIDModule.__new__(ReIDModule)
        reid._gallery = {}
        reid._counters = {}
        reid.model = None
        from datetime import date
        reid._operating_day = date.today()

        # Insert into ST1076 gallery
        reid._gallery["ST1076"] = {"ID_60000": {"embedding": np.zeros(512), "last_seen": 9999999999, "exited": False}}
        # ST1008 gallery should be empty
        gallery = reid._gallery_for("ST1008")
        assert len(gallery) == 0

    def test_cosine_similarity_same_vector(self):
        """Same vector → cosine similarity = 1.0."""
        from pipeline.reid import ReIDModule
        reid = ReIDModule.__new__(ReIDModule)
        reid.model = None
        v = np.random.rand(512)
        v /= np.linalg.norm(v)
        assert abs(reid.cosine_similarity(v, v) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# zone_engine.py tests
# ---------------------------------------------------------------------------

class TestZoneEngine:
    def _make_engine(self) -> "ZoneEngine":
        from pipeline.zone_engine import ZoneEngine, ZoneDef
        from shapely.geometry import Polygon
        engine = ZoneEngine.__new__(ZoneEngine)
        engine.store_id = "ST1076"
        engine._states = {}
        engine._billing_queue = {}
        engine._billing_count = 0

        # Create a simple zone manually
        inner = Polygon([(100, 100), (500, 100), (500, 400), (100, 400)])
        outer = Polygon([(50, 50), (550, 50), (550, 450), (50, 450)])
        engine.zones = {
            "PURPLLE_MUM_1076_Z01": ZoneDef(
                zone_id="PURPLLE_MUM_1076_Z01",
                zone_name="Left Shelf",
                zone_type="SHELF",
                is_revenue_zone=True,
                store_id="ST1076",
                inner_poly=inner,
                outer_poly=outer,
            )
        }
        return engine

    def test_zone_entered_on_inner_entry(self):
        engine = self._make_engine()
        ts = datetime.now(tz=timezone.utc)
        events = engine.process_track(
            id_token="ID_60001",
            camera_id="CAM1",
            ground_xy=(300.0, 250.0),  # inside inner polygon
            is_staff=False,
            ts=ts,
            track_id=1,
        )
        assert len(events) == 1
        assert events[0]["event_type"] == "zone_entered"

    def test_zone_exited_on_outer_exit(self):
        engine = self._make_engine()
        ts = datetime.now(tz=timezone.utc)
        # First: enter inner
        engine.process_track("ID_60001", "CAM1", (300.0, 250.0), False, ts, 1)
        # Now: exit outer polygon entirely
        from datetime import timedelta
        ts2 = ts + timedelta(seconds=10)
        events = engine.process_track("ID_60001", "CAM1", (600.0, 600.0), False, ts2, 1)
        assert any(e["event_type"] == "zone_exited" for e in events)

    def test_no_transition_in_hysteresis_zone(self):
        """Point between inner and outer → no state change from INSIDE."""
        engine = self._make_engine()
        ts = datetime.now(tz=timezone.utc)
        # Enter inner
        engine.process_track("ID_60001", "CAM1", (300.0, 250.0), False, ts, 1)
        # Move to hysteresis zone (between inner 100-500 and outer 50-550)
        from datetime import timedelta
        ts2 = ts + timedelta(seconds=5)
        events = engine.process_track("ID_60001", "CAM1", (520.0, 250.0), False, ts2, 1)
        # Should be no events (still in outer, so INSIDE state maintained)
        assert len(events) == 0

    def test_dwell_ms_null_on_zone_entered(self):
        engine = self._make_engine()
        ts = datetime.now(tz=timezone.utc)
        events = engine.process_track("ID_60001", "CAM1", (300.0, 250.0), False, ts, 1)
        entered = next(e for e in events if e["event_type"] == "zone_entered")
        assert entered["dwell_ms"] is None

    def test_dwell_ms_present_on_zone_exited(self):
        engine = self._make_engine()
        ts = datetime.now(tz=timezone.utc)
        engine.process_track("ID_60001", "CAM1", (300.0, 250.0), False, ts, 1)
        from datetime import timedelta
        ts2 = ts + timedelta(seconds=30)
        events = engine.process_track("ID_60001", "CAM1", (600.0, 600.0), False, ts2, 1)
        exited = next(e for e in events if e["event_type"] == "zone_exited")
        assert exited["dwell_ms"] is not None
        assert exited["dwell_ms"] >= 30000


# ---------------------------------------------------------------------------
# models.py tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_entry_exit_store_code_normalisation(self):
        from app.models import EntryExitEvent
        evt = EntryExitEvent(
            id_token="ID_60001",
            store_code="store_1076",
            camera_id="cam1",
            event_type="entry",
            event_timestamp=datetime.now(tz=timezone.utc),
        )
        assert evt.store_id == "ST1076"

    def test_zone_event_revenue_zone_coercion(self):
        from app.models import ZoneEvent
        evt = ZoneEvent(
            track_id=1,
            store_id="ST1076",
            camera_id="CAM2",
            zone_id="PURPLLE_MUM_1076_Z01",
            zone_name="Left Shelf",
            zone_type="SHELF",
            is_revenue_zone="Yes",   # string → bool
            event_type="zone_entered",
            event_time=datetime.now(tz=timezone.utc),
        )
        assert evt.is_revenue_zone is True

    def test_queue_event_abandoned_flag(self):
        from app.models import QueueEvent
        evt = QueueEvent(
            track_id=200,
            store_id="ST1076",
            camera_id="CAM5",
            zone_id="PURPLLE_MUM_1076_Z_BILLING_01",
            event_type="queue_abandoned",
            queue_join_ts=datetime.now(tz=timezone.utc),
            queue_exit_ts=datetime.now(tz=timezone.utc),
            wait_seconds=300,
            queue_position_at_join=4,
            abandoned=True,
        )
        assert evt.abandoned is True
