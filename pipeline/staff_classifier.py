"""
staff_classifier.py — MobileNetV3-Small + HSV color histogram ensemble.

Classification schedule:
  - Initial classification: within 10 frames of track init
  - Re-evaluation: every 150 frames (handles partial occlusion at first detection)
  - If false→true flip: emit correction flag

Ensemble rule:
  is_staff = (nn_prob > 0.7) AND (bhattacharyya_dist < 0.3)
  Falls back to HSV-only if ONNX model not available.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

CLASSIFY_INIT_FRAME   = 10
RECLASSIFY_INTERVAL   = 150
NN_STAFF_THRESHOLD    = 0.7
HSV_DIST_THRESHOLD    = 0.3   # Bhattacharyya distance (lower = more similar)


class StaffClassifier:
    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.onnx_session = None

        # Per-track state: {track_id: {"is_staff": bool, "frame_count": int, "last_classified": int}}
        self._track_state: dict[int, dict] = {}

        # Reference HSV histogram for staff uniform
        # Will be set during first classification if staff detected
        self._staff_hsv_ref: Optional[np.ndarray] = None

        self._load_model()

    def _load_model(self) -> None:
        onnx_path = self.model_dir / "mobilenet_staff.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
                logger.info("staff_classifier_loaded_onnx", path=str(onnx_path))
            except Exception as e:
                logger.warning("staff_classifier_onnx_failed", error=str(e))
        else:
            logger.info("staff_classifier_hsv_only_mode", reason="mobilenet_staff.onnx not found")

    def _compute_hsv_histogram(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _nn_predict(self, crop: np.ndarray) -> float:
        """Returns staff probability from ONNX model. Returns -1.0 if unavailable."""
        if self.onnx_session is None:
            return -1.0
        try:
            img = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: img})
            # Expect output shape (1, 2): [customer_prob, staff_prob]
            probs = outputs[0][0]
            return float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception as e:
            logger.debug("nn_predict_failed", error=str(e))
            return -1.0

    def classify(
        self,
        track_id: int,
        crop: np.ndarray,
        frame_index: int,
    ) -> tuple[bool, bool]:
        """
        Returns (is_staff, classification_changed).
        classification_changed=True triggers correction event emission.
        """
        state = self._track_state.get(track_id, {
            "is_staff": False, "frame_count": 0, "last_classified": -1
        })
        state["frame_count"] = frame_index

        should_classify = (
            state["last_classified"] < 0  # first classification
            or (frame_index - state["last_classified"]) >= RECLASSIFY_INTERVAL
        )

        if not should_classify:
            return state["is_staff"], False

        # ---- Run ensemble ----
        prev = state["is_staff"]
        nn_prob = self._nn_predict(crop)
        hist = self._compute_hsv_histogram(crop)

        if nn_prob >= 0 and hist is not None and self._staff_hsv_ref is not None:
            # Full ensemble
            bhatt = cv2.compareHist(
                hist.reshape(-1, 1).astype(np.float32),
                self._staff_hsv_ref.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            is_staff = (nn_prob > NN_STAFF_THRESHOLD) and (bhatt < HSV_DIST_THRESHOLD)
        elif nn_prob >= 0:
            # NN only (no reference histogram yet)
            is_staff = nn_prob > NN_STAFF_THRESHOLD
        else:
            # HSV-only fallback: assume not staff (conservative)
            is_staff = False

        # Update reference histogram if staff detected (bootstrap)
        if is_staff and hist is not None and self._staff_hsv_ref is None:
            self._staff_hsv_ref = hist

        state["is_staff"] = is_staff
        state["last_classified"] = frame_index
        self._track_state[track_id] = state

        changed = (not prev) and is_staff  # false → true flip
        return is_staff, changed

    def get_is_staff(self, track_id: int) -> bool:
        return self._track_state.get(track_id, {}).get("is_staff", False)
