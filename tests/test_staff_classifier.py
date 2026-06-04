from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import cv2
from pipeline.staff_classifier import StaffClassifier, RECLASSIFY_INTERVAL

@pytest.fixture
def classifier():
    return StaffClassifier(model_dir="non_existent_dir")

def test_init_no_model(classifier):
    assert classifier.onnx_session is None

def test_compute_hsv_histogram(classifier):
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = [0, 255, 0]  # Green image
    hist = classifier._compute_hsv_histogram(crop)
    assert hist is not None
    assert hist.shape == (36 * 32,)

def test_compute_hsv_histogram_empty(classifier):
    assert classifier._compute_hsv_histogram(np.array([])) is None
    assert classifier._compute_hsv_histogram(None) is None

def test_nn_predict_no_model(classifier):
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    prob = classifier._nn_predict(crop)
    assert prob == -1.0

def test_classify_hsv_fallback(classifier):
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    is_staff, changed = classifier.classify(track_id=1, crop=crop, frame_index=0)
    # Without model, fallback is False
    assert is_staff is False
    assert changed is False

def test_classify_skips_until_interval(classifier):
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    classifier.classify(1, crop, 0)
    
    # Mock compute_hsv_histogram to verify it's not called
    with patch.object(classifier, '_compute_hsv_histogram') as mock_hist:
        # Before interval
        classifier.classify(1, crop, 50)
        mock_hist.assert_not_called()
        
        # After interval
        classifier.classify(1, crop, RECLASSIFY_INTERVAL + 1)
        mock_hist.assert_called_once()

def test_get_is_staff(classifier):
    assert classifier.get_is_staff(999) is False
    classifier._track_state[1] = {"is_staff": True}
    assert classifier.get_is_staff(1) is True

def test_classify_ensemble_flip():
    # Test that flip triggers changed=True
    classifier = StaffClassifier(model_dir="non_existent_dir")
    
    with patch.object(classifier, '_nn_predict', return_value=0.9):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        # First classification -> NN prob is 0.9 (>0.7), no ref hist -> is_staff=True
        is_staff, changed = classifier.classify(1, crop, 0)
        assert is_staff is True
        assert changed is True  # False -> True flip
        
        # Verify reference histogram was set
        assert classifier._staff_hsv_ref is not None
