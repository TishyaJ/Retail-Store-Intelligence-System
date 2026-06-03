"""
main_pipeline.py — Per-camera CLI entry point.

Usage:
  python main_pipeline.py \\
    --clip "Stores/Store 1/CAM 1 - zone.mp4" \\
    --store_id ST1076 \\
    --camera_id CAM1 \\
    --camera_type zone

Orchestrates the full pipeline chain per frame:
  VideoCapture → detect → track → reid → staff_classify → homography → zone_engine → emit
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import structlog
from dotenv import load_dotenv

load_dotenv()

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-camera vision pipeline")
    p.add_argument("--clip",        required=True,  help="Path to video clip file")
    p.add_argument("--store_id",    required=True,  help="e.g. ST1076")
    p.add_argument("--camera_id",   required=True,  help="e.g. CAM1")
    p.add_argument("--camera_type", required=True,  choices=["entry", "zone", "billing"],
                   help="Camera function type")
    return p.parse_args()


def load_clip_config(camera_id: str) -> dict:
    """Load camera start_utc and calibration from camera_config.json."""
    config_path = os.environ.get("CAMERA_CONFIG_PATH", "store_layout/camera_config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        for clip in config.get("clips", []):
            if clip["camera_id"] == camera_id:
                return clip
    except Exception as e:
        logger.warning("camera_config_load_failed", error=str(e))
    return {}


def main() -> None:
    args = parse_args()
    api_url = os.environ.get("API_SERVER_URL", "http://api-server:8000")

    clip_config = load_clip_config(args.camera_id)
    start_utc_str = clip_config.get("start_utc") or os.environ.get(
        f"STORE{'1' if '1076' in args.store_id else '2'}_CLIP_START_UTC",
        "2026-03-08T12:00:00Z",
    )
    clip_start_utc = datetime.fromisoformat(start_utc_str.replace("Z", "+00:00"))
    fps = clip_config.get("fps", 15)

    # Bind context for all log lines
    structlog.contextvars.bind_contextvars(
        service="vision-pipeline",
        store_id=args.store_id,
        camera_id=args.camera_id,
        camera_type=args.camera_type,
    )

    logger.info("pipeline_starting", clip=args.clip, start_utc=start_utc_str)

    # ---- Lazy imports (heavy) ----
    from .detect         import Detector
    from .tracker        import ByteTracker
    from .reid           import ReIDModule
    from .staff_classifier import StaffClassifier
    from .homography     import CrossCameraDeduplicator
    from .zone_engine    import ZoneEngine
    from .emit           import EventEmitter

    detector    = Detector(model_dir="models")
    tracker     = ByteTracker()
    reid_module = ReIDModule(model_dir="models")
    classifier  = StaffClassifier(model_dir="models")
    dedup       = CrossCameraDeduplicator(config_path="store_layout/camera_config.json")
    zone_engine = ZoneEngine(zones_config_path="store_layout/zones.json", store_id=args.store_id)
    emitter     = EventEmitter(api_server_url=api_url, store_id=args.store_id, camera_id=args.camera_id)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        logger.critical("cannot_open_clip", clip=args.clip)
        sys.exit(1)

    # ---- Graceful shutdown ----
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    frame_index = 0
    frames_processed = 0
    detections_total = 0
    events_total = 0
    last_log_time = time.monotonic()

    store_code = f"store_{args.store_id[2:].lower()}"  # ST1076 → store_1076

    while running:
        ret, frame = cap.read()
        if not ret:
            break  # end of clip

        ts: datetime = clip_start_utc + timedelta(seconds=frame_index / fps)
        frame_index += 1
        frames_processed += 1

        # ---- Detect ----
        try:
            high_dets, low_dets = detector.detect(frame, args.camera_id, frame_index)
        except Exception as e:
            logger.warning("detect_error", error=str(e), frame_index=frame_index)
            continue

        detections_total += len(high_dets) + len(low_dets)

        # ---- Track ----
        active_tracks = tracker.update(high_dets, low_dets)

        for track in active_tracks:
            # Crop for Re-ID and classification
            x1, y1, x2, y2 = [int(c) for c in track.bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None

            # ---- Re-ID ----
            embedding = reid_module.extract_embedding(crop) if crop is not None else None

            if track.id_token is None:
                id_token, is_reentry = reid_module.lookup_or_create(args.store_id, embedding)
                track.id_token = id_token
                track.embedding = embedding

                # Emit ENTRY/REENTRY for entry-camera type
                if args.camera_type == "entry":
                    evt = emitter.build_entry_exit_event(
                        id_token=id_token,
                        direction="reentry" if is_reentry else "entry",
                        store_code=store_code,
                        store_id=args.store_id,
                        camera_id=args.camera_id,
                        ts=ts,
                        confidence=track.confidence,
                    )
                    emitter.push(evt)
                    events_total += 1
            else:
                track.embedding = embedding

            # ---- Staff Classification ----
            if crop is not None:
                is_staff, changed = classifier.classify(track.track_id, crop, frame_index)
                track.is_staff = is_staff

            # ---- Cross-Camera Deduplication ----
            H = dedup.H.get(args.camera_id)
            if H is not None:
                from .homography import project_to_ground
                gx, gy = project_to_ground(track.foot_point, H)
                merged_token = dedup.update(
                    camera_id=args.camera_id,
                    track_id=track.track_id,
                    foot_pixel=track.foot_point,
                    embedding=embedding,
                    id_token=track.id_token,
                )
                if merged_token != track.id_token:
                    track.id_token = merged_token
                ground_xy = (gx, gy)
            else:
                # No homography available — use pixel foot-point as fallback
                ground_xy = track.foot_point

            # ---- Zone Engine ----
            if args.camera_type in ("zone", "billing"):
                zone_events = zone_engine.process_track(
                    id_token=track.id_token,
                    camera_id=args.camera_id,
                    ground_xy=ground_xy,
                    is_staff=track.is_staff,
                    ts=ts,
                    track_id=track.track_id,
                )
                for evt in zone_events:
                    emitter.push(evt)
                    events_total += 1

        # ---- Periodic stats log ----
        now = time.monotonic()
        if now - last_log_time >= 60:
            logger.info(
                "pipeline_stats",
                frames=frames_processed,
                detections=detections_total,
                events=events_total,
                frame_index=frame_index,
            )
            last_log_time = now

    # ---- Cleanup ----
    cap.release()
    emitter.flush()
    logger.info("pipeline_finished", frames=frames_processed, events=events_total)


if __name__ == "__main__":
    main()
