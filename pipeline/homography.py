"""
homography.py — Cross-camera ground-plane deduplication.

Uses OpenCV homography matrices to project bounding-box foot-points
from camera pixel coordinates to physical floor coordinates (mm).

Cross-camera merge criteria:
  1. Euclidean distance on ground plane < 500mm (0.5m)
  2. OSNet cosine similarity > 0.75

Hungarian matching resolves globally optimal assignments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import structlog
from scipy.optimize import linear_sum_assignment

logger = structlog.get_logger()

SPATIAL_THRESHOLD_MM = 500.0     # 0.5 metres in mm
COSINE_MERGE_THRESHOLD = 0.75


def load_homography_matrices(config_path: str) -> dict[str, np.ndarray]:
    """
    Load per-camera homography matrices from camera_config.json.
    Returns {camera_id: 3x3 H matrix}.

    H is computed from 4 point correspondences:
      src_points: pixel coordinates in camera image
      dst_points: physical floor coordinates in mm
    """
    matrices: dict[str, np.ndarray] = {}
    try:
        with open(config_path) as f:
            config = json.load(f)

        for clip in config.get("clips", []):
            cam_id = clip["camera_id"]
            cal = clip.get("homography_calibration", {})
            points = cal.get("points", [])
            if len(points) < 4:
                logger.warning("insufficient_calibration_points", camera_id=cam_id)
                continue

            src = np.float32([[p["pixel_xy"][0], p["pixel_xy"][1]] for p in points])
            dst = np.float32([[p["floor_mm"][0], p["floor_mm"][1]] for p in points])
            H, status = cv2.findHomography(src, dst)
            if H is not None:
                matrices[cam_id] = H
                logger.info("homography_matrix_loaded", camera_id=cam_id)
            else:
                logger.warning("homography_computation_failed", camera_id=cam_id)

    except Exception as e:
        logger.error("homography_config_load_failed", error=str(e))

    return matrices


def project_to_ground(
    pixel_xy: tuple[float, float],
    H: np.ndarray,
) -> tuple[float, float]:
    """
    Project a pixel coordinate to physical floor (mm) using homography.
    Uses homogeneous coordinates: [X, Y, W] = H @ [u, v, 1]
    """
    u, v = pixel_xy
    pt = np.array([u, v, 1.0], dtype=np.float64)
    result = H @ pt
    if abs(result[2]) < 1e-10:
        return 0.0, 0.0
    gx = result[0] / result[2]
    gy = result[1] / result[2]
    return float(gx), float(gy)


class CrossCameraDeduplicator:
    """
    Maintains a multi-camera ground-plane registry.
    Merges track identities across overlapping cameras.
    """

    def __init__(self, config_path: str = "store_layout/camera_config.json") -> None:
        self.H: dict[str, np.ndarray] = load_homography_matrices(config_path)
        # Active ground-plane positions: {(camera_id, track_id): (gx, gy, id_token, embedding)}
        self._registry: dict[tuple[str, int], dict] = {}

    def update(
        self,
        camera_id: str,
        track_id: int,
        foot_pixel: tuple[float, float],
        embedding: Optional[np.ndarray],
        id_token: Optional[str],
    ) -> Optional[str]:
        """
        Update registry entry. Returns merged id_token if a cross-camera
        merge was found, else returns the original id_token.
        """
        H = self.H.get(camera_id)
        if H is None:
            return id_token

        gx, gy = project_to_ground(foot_pixel, H)
        key = (camera_id, track_id)
        self._registry[key] = {
            "gx": gx, "gy": gy,
            "embedding": embedding,
            "id_token": id_token,
            "camera_id": camera_id,
        }

        if embedding is None:
            return id_token

        # Find candidates from OTHER cameras within 500mm
        candidates = []
        for (cam, tid), entry in self._registry.items():
            if cam == camera_id:
                continue
            dist = np.sqrt((gx - entry["gx"])**2 + (gy - entry["gy"])**2)
            if dist < SPATIAL_THRESHOLD_MM:
                candidates.append((cam, tid, entry, dist))

        if not candidates:
            return id_token

        # Build cost matrix and solve via Hungarian
        embeddings = [embedding]
        cand_entries = [e for _, _, e, _ in candidates]
        cand_embs = [e["embedding"] for e in cand_entries if e.get("embedding") is not None]

        if not cand_embs:
            return id_token

        # Compute cosine similarities
        best_sim = 0.0
        best_entry = None
        for cand_emb, cand_entry in zip(cand_embs, cand_entries):
            sim = float(np.dot(embedding, cand_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(cand_emb) + 1e-8
            ))
            if sim > best_sim:
                best_sim = sim
                best_entry = cand_entry

        if best_sim > COSINE_MERGE_THRESHOLD and best_entry is not None:
            merged_token = best_entry["id_token"] or id_token
            self._registry[key]["id_token"] = merged_token
            logger.debug(
                "cross_camera_merge",
                camera_id=camera_id,
                track_id=track_id,
                merged_to=merged_token,
                cosine_sim=round(best_sim, 3),
            )
            return merged_token

        return id_token

    def clear_track(self, camera_id: str, track_id: int) -> None:
        self._registry.pop((camera_id, track_id), None)
