"""
detect.py — YOLOv11m person detection using ONNX or ultralytics fallback.

Model strategy:
  - Primary: load ONNX model from models/yolov11m_retail.onnx (custom trained)
  - Fallback: use ultralytics pretrained YOLOv11m (if ONNX not available)
  - Swappable: drop a new .onnx file in models/ and restart — no code changes needed

ByteTrack confidence split:
  HIGH_CONF_THRESHOLD = 0.60   first-pass matching
  LOW_CONF_THRESHOLD  = 0.30   second-pass matching (not discarded)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

HIGH_CONF_THRESHOLD = 0.60
LOW_CONF_THRESHOLD = 0.30
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = PERSON_CLASS_ID

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of bounding box — used for ground-plane projection."""
        return (self.x1 + self.x2) / 2, self.y2


class Detector:
    """
    YOLOv11m person detector.

    Falls back to ultralytics pretrained model if ONNX not available.
    """

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.onnx_session = None
        self.yolo_model = None
        self._load_model()

    def _load_model(self) -> None:
        onnx_path = self.model_dir / "yolov11m_retail.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                logger.info("detector_loaded_onnx", path=str(onnx_path))
                return
            except Exception as e:
                logger.warning("onnx_load_failed", error=str(e))

        # Fallback: ultralytics pretrained YOLOv11m
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolo11m.pt")
            logger.info("detector_loaded_ultralytics_pretrained")
        except Exception as e:
            logger.critical("detector_load_failed", error=str(e))
            raise

    def detect(
        self, frame: np.ndarray, camera_id: str = "", frame_index: int = 0
    ) -> tuple[list[Detection], list[Detection]]:
        """
        Run detection on a BGR frame.

        Returns:
            high_conf: detections with confidence > HIGH_CONF_THRESHOLD
            low_conf:  detections with LOW_CONF_THRESHOLD < conf ≤ HIGH_CONF_THRESHOLD
        """
        if frame is None or frame.size == 0:
            logger.warning("frame_decode_failure", camera_id=camera_id, frame_index=frame_index)
            return [], []

        if self.onnx_session is not None:
            return self._detect_onnx(frame)
        return self._detect_ultralytics(frame)

    def _detect_onnx(
        self, frame: np.ndarray
    ) -> tuple[list[Detection], list[Detection]]:
        # Preprocess: BGR → RGB, resize, normalise
        input_name = self.onnx_session.get_inputs()[0].name
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :]

        outputs = self.onnx_session.run(None, {input_name: img})
        # outputs[0] shape: (1, num_preds, 6) — [x1,y1,x2,y2,conf,cls]
        preds = outputs[0][0]

        h, w = frame.shape[:2]
        sx, sy = w / 640, h / 640

        return self._split_detections(preds, sx, sy)

    def _detect_ultralytics(
        self, frame: np.ndarray
    ) -> tuple[list[Detection], list[Detection]]:
        results = self.yolo_model(frame, conf=LOW_CONF_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls != PERSON_CLASS_ID or conf < LOW_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf))

        high = [d for d in detections if d.confidence > HIGH_CONF_THRESHOLD]
        low  = [d for d in detections if LOW_CONF_THRESHOLD <= d.confidence <= HIGH_CONF_THRESHOLD]
        return high, low

    def _split_detections(
        self, preds: np.ndarray, sx: float, sy: float
    ) -> tuple[list[Detection], list[Detection]]:
        high, low = [], []
        for pred in preds:
            conf = float(pred[4])
            cls  = int(pred[5])
            if cls != PERSON_CLASS_ID or conf < LOW_CONF_THRESHOLD:
                continue
            d = Detection(
                x1=float(pred[0]) * sx,
                y1=float(pred[1]) * sy,
                x2=float(pred[2]) * sx,
                y2=float(pred[3]) * sy,
                confidence=conf,
            )
            if conf > HIGH_CONF_THRESHOLD:
                high.append(d)
            else:
                low.append(d)
        return high, low
