"""
reid.py — OSNet Re-ID appearance gallery.

Uses torchreid pretrained OSNet-x0_25 (auto-downloaded from model zoo).
Maintains per-store, per-day galleries keyed by id_token.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
import torch

logger = structlog.get_logger()

COSINE_REENTRY_THRESHOLD  = 0.85   # similarity > this → same person re-entering
COSINE_DEDUP_THRESHOLD    = 0.75   # used by homography.py for cross-camera dedup
GALLERY_TTL_HOURS         = 12

# Per-store id_token base offsets (prevent cross-store collisions)
STORE_ID_OFFSETS: dict[str, int] = {
    "ST1076": 60000,
    "ST1008": 70000,
}
DEFAULT_ID_OFFSET = 80000


class ReIDModule:
    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.model = None
        self.transform = None

        # Gallery: {store_id: {id_token: {"embedding": np.ndarray, "last_seen": float}}}
        self._gallery: dict[str, dict[str, dict]] = {}

        # Per-store counters for id_token generation
        self._counters: dict[str, int] = {}

        # Current operating day
        self._operating_day = date.today()

        self._load_model()

    def _load_model(self) -> None:
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name="osnet_x0_25", num_classes=1000, pretrained=True
            )
            self.model.eval()
            self.model.classifier = torch.nn.Identity()  # remove classifier head
            logger.info("reid_model_loaded", model="osnet_x0_25")
        except Exception as e:
            logger.warning("reid_model_load_failed", error=str(e))
            self.model = None

    def _get_transform(self):
        if self.transform is not None:
            return self.transform
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return self.transform

    def extract_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-dim L2-normalised embedding from a BGR crop."""
        if self.model is None or crop is None or crop.size == 0:
            return None
        try:
            import cv2
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            transform = self._get_transform()
            tensor = transform(rgb).unsqueeze(0)
            with torch.no_grad():
                feat = self.model(tensor)
            feat = feat.squeeze().numpy()
            norm = np.linalg.norm(feat)
            return feat / norm if norm > 0 else feat
        except Exception as e:
            logger.debug("embedding_extraction_failed", error=str(e))
            return None

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _reset_if_new_day(self, store_id: str) -> None:
        today = date.today()
        if today != self._operating_day:
            self._gallery = {}
            self._counters = {}
            self._operating_day = today

    def _gallery_for(self, store_id: str) -> dict:
        if store_id not in self._gallery:
            self._gallery[store_id] = {}
        return self._gallery[store_id]

    def _next_id_token(self, store_id: str) -> str:
        base = STORE_ID_OFFSETS.get(store_id, DEFAULT_ID_OFFSET)
        count = self._counters.get(store_id, 0)
        self._counters[store_id] = count + 1
        return f"ID_{base + count}"

    def _evict_expired(self, gallery: dict) -> None:
        now = time.time()
        expired = [k for k, v in gallery.items() if now - v["last_seen"] > GALLERY_TTL_HOURS * 3600]
        for k in expired:
            del gallery[k]

    def lookup_or_create(
        self,
        store_id: str,
        embedding: Optional[np.ndarray],
    ) -> tuple[str, bool]:
        """
        Returns (id_token, is_reentry).
        is_reentry=True if this visitor was previously seen (EXIT recorded).
        """
        self._reset_if_new_day(store_id)
        gallery = self._gallery_for(store_id)
        self._evict_expired(gallery)

        if embedding is not None:
            best_id, best_sim = None, 0.0
            for token, entry in gallery.items():
                sim = self.cosine_similarity(embedding, entry["embedding"])
                if sim > best_sim:
                    best_sim, best_id = sim, token

            if best_sim > COSINE_REENTRY_THRESHOLD and best_id is not None:
                gallery[best_id]["last_seen"] = time.time()
                gallery[best_id]["embedding"] = embedding  # update gallery
                return best_id, gallery[best_id].get("exited", False)

        # New visitor
        token = self._next_id_token(store_id)
        gallery[token] = {
            "embedding":  embedding if embedding is not None else np.zeros(512),
            "last_seen":  time.time(),
            "exited":     False,
        }
        return token, False

    def mark_exited(self, store_id: str, id_token: str) -> None:
        gallery = self._gallery_for(store_id)
        if id_token in gallery:
            gallery[id_token]["exited"] = True

    def get_embedding(self, store_id: str, id_token: str) -> Optional[np.ndarray]:
        gallery = self._gallery_for(store_id)
        entry = gallery.get(id_token)
        return entry["embedding"] if entry else None
