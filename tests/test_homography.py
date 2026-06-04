from __future__ import annotations

import json
from unittest.mock import patch, mock_open
import numpy as np

import pytest
from pipeline.homography import (
    load_homography_matrices,
    project_to_ground,
    CrossCameraDeduplicator,
    SPATIAL_THRESHOLD_MM,
)

def test_load_homography_matrices_valid():
    config_data = {
        "clips": [
            {
                "camera_id": "CAM1",
                "homography_calibration": {
                    "points": [
                        {"pixel_xy": [0, 0], "floor_mm": [0, 0]},
                        {"pixel_xy": [100, 0], "floor_mm": [1000, 0]},
                        {"pixel_xy": [100, 100], "floor_mm": [1000, 1000]},
                        {"pixel_xy": [0, 100], "floor_mm": [0, 1000]},
                    ]
                }
            }
        ]
    }
    with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
        matrices = load_homography_matrices("dummy.json")
        assert "CAM1" in matrices
        assert matrices["CAM1"].shape == (3, 3)

def test_load_homography_matrices_invalid():
    with patch("builtins.open", side_effect=FileNotFoundError):
        matrices = load_homography_matrices("missing.json")
        assert matrices == {}

def test_project_to_ground():
    H = np.eye(3)
    gx, gy = project_to_ground((150.0, 200.0), H)
    assert gx == 150.0
    assert gy == 200.0

def test_project_to_ground_zero_division():
    H = np.zeros((3, 3))
    gx, gy = project_to_ground((150.0, 200.0), H)
    assert gx == 0.0
    assert gy == 0.0

class TestCrossCameraDeduplicator:
    @pytest.fixture
    def dedup(self):
        with patch("pipeline.homography.load_homography_matrices", return_value={"CAM1": np.eye(3), "CAM2": np.eye(3)}):
            return CrossCameraDeduplicator("dummy.json")

    def test_update_no_homography(self, dedup):
        token = dedup.update("CAM_UNKNOWN", 1, (100, 100), None, "ID_1")
        assert token == "ID_1"

    def test_update_no_embedding(self, dedup):
        token = dedup.update("CAM1", 1, (100, 100), None, "ID_1")
        assert token == "ID_1"
        assert ("CAM1", 1) in dedup._registry

    def test_update_merge_successful(self, dedup):
        emb1 = np.ones(512)
        emb1 /= np.linalg.norm(emb1)
        dedup.update("CAM1", 1, (100, 100), emb1, "ID_MAIN")
        
        # CAM2 same position, same embedding
        token = dedup.update("CAM2", 2, (110, 110), emb1, "ID_NEW")
        assert token == "ID_MAIN"

    def test_update_no_merge_spatial_dist_too_large(self, dedup):
        emb1 = np.ones(512)
        emb1 /= np.linalg.norm(emb1)
        dedup.update("CAM1", 1, (0, 0), emb1, "ID_MAIN")
        
        # CAM2 too far
        token = dedup.update("CAM2", 2, (1000, 1000), emb1, "ID_NEW")
        assert token == "ID_NEW"

    def test_clear_track(self, dedup):
        dedup.update("CAM1", 1, (100, 100), None, "ID_1")
        assert ("CAM1", 1) in dedup._registry
        dedup.clear_track("CAM1", 1)
        assert ("CAM1", 1) not in dedup._registry
