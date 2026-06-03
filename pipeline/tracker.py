"""
tracker.py — ByteTrack multi-object tracker.

Dual-pass Hungarian matching:
  1st pass: TRACKED tracks ↔ high-conf detections (IoU)
  2nd pass: TRACKED+LOST tracks ↔ low-conf detections (IoU)
  Remaining high-conf → new NEW tracks

Track lifecycle: NEW → TRACKED → LOST → REMOVED (30-frame threshold)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection

LOST_THRESHOLD = 30
NEW_CONFIRM_FRAMES = 3


class TrackState(str, Enum):
    NEW     = "new"
    TRACKED = "tracked"
    LOST    = "lost"
    REMOVED = "removed"


@dataclass
class KalmanState:
    """Simple Kalman filter state [cx, cy, area, aspect, vx, vy, va]."""
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class Track:
    track_id: int
    state: TrackState
    bbox: tuple[float, float, float, float]     # x1,y1,x2,y2
    confidence: float
    age: int = 0                                 # frames since creation
    lost_frames: int = 0                         # consecutive lost frames
    confirmed_frames: int = 0                    # consecutive matched frames

    # Re-ID embedding (set by reid.py)
    embedding: Optional[np.ndarray] = None

    # Identity (set by reid.py / homography.py)
    id_token: Optional[str] = None
    is_staff:  bool = False

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, y2


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def hungarian_match(
    tracks: list[Track],
    detections: list[Detection],
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Returns:
        matches:        list of (track_idx, det_idx) pairs
        unmatched_tracks: track indices with no match
        unmatched_dets:   detection indices with no match
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.zeros((len(tracks), len(detections)))
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            cost[ti, di] = 1.0 - iou(t.bbox, d.bbox)

    row_idx, col_idx = linear_sum_assignment(cost)

    matches, unmatched_t, unmatched_d = [], [], []
    matched_t, matched_d = set(), set()

    for ri, ci in zip(row_idx, col_idx):
        if cost[ri, ci] <= 1.0 - iou_threshold:
            matches.append((ri, ci))
            matched_t.add(ri)
            matched_d.add(ci)

    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
    return matches, unmatched_t, unmatched_d


_track_counter = 0


def _next_track_id() -> int:
    global _track_counter
    _track_counter += 1
    return _track_counter


class ByteTracker:
    def __init__(self) -> None:
        self.tracks: list[Track] = []

    def update(
        self,
        high_dets: list[Detection],
        low_dets:  list[Detection],
    ) -> list[Track]:
        """Update tracker state for one frame. Returns currently active tracks."""
        tracked = [t for t in self.tracks if t.state in (TrackState.TRACKED, TrackState.NEW)]
        lost    = [t for t in self.tracks if t.state == TrackState.LOST]

        # ---- Pass 1: tracked ↔ high-conf detections ----
        matches1, unmatched_t1, unmatched_d1 = hungarian_match(tracked, high_dets)

        for ti, di in matches1:
            t = tracked[ti]
            d = high_dets[di]
            t.bbox = d.bbox
            t.confidence = d.confidence
            t.lost_frames = 0
            t.confirmed_frames += 1
            t.age += 1
            if t.confirmed_frames >= NEW_CONFIRM_FRAMES:
                t.state = TrackState.TRACKED

        unmatched_tracked = [tracked[i] for i in unmatched_t1]
        unmatched_high    = [high_dets[i] for i in unmatched_d1]

        # ---- Pass 2: unmatched tracked + lost ↔ low-conf detections ----
        pass2_tracks = unmatched_tracked + lost
        matches2, unmatched_t2, _ = hungarian_match(pass2_tracks, low_dets, iou_threshold=0.2)

        for ti, di in matches2:
            t = pass2_tracks[ti]
            d = low_dets[di]
            t.bbox = d.bbox
            t.confidence = d.confidence
            t.lost_frames = 0
            t.confirmed_frames += 1
            t.age += 1
            t.state = TrackState.TRACKED

        # ---- Tracks with no match → LOST ----
        for ti in unmatched_t2:
            t = pass2_tracks[ti]
            t.lost_frames += 1
            t.age += 1
            if t.lost_frames >= LOST_THRESHOLD:
                t.state = TrackState.REMOVED
            else:
                t.state = TrackState.LOST

        # ---- New tracks from unmatched high-conf detections ----
        for d in unmatched_high:
            self.tracks.append(Track(
                track_id=_next_track_id(),
                state=TrackState.NEW,
                bbox=d.bbox,
                confidence=d.confidence,
            ))

        # ---- Prune removed tracks ----
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        return [t for t in self.tracks if t.state in (TrackState.TRACKED, TrackState.NEW)]
