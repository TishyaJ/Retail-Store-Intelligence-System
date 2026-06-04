from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from pipeline.detect import Detector, Detection


@pytest.fixture
def empty_detector():
    with patch("pipeline.detect.Path.exists", return_value=False):
        with patch("ultralytics.YOLO") as MockYOLO:
            yield Detector(model_dir="non_existent")


def test_detector_load_onnx():
    with patch("pipeline.detect.Path.exists", return_value=True):
        with patch("onnxruntime.InferenceSession") as MockOrt:
            d = Detector(model_dir="dummy")
            assert d.onnx_session is not None
            MockOrt.assert_called_once()


def test_detector_load_ultralytics():
    with patch("pipeline.detect.Path.exists", return_value=False):
        with patch("ultralytics.YOLO") as MockYOLO:
            d = Detector(model_dir="dummy")
            assert d.yolo_model is not None
            MockYOLO.assert_called_once()


def test_detector_load_ultralytics_failure():
    with patch("pipeline.detect.Path.exists", return_value=False):
        with patch("ultralytics.YOLO", side_effect=Exception("mock fail")):
            with pytest.raises(Exception):
                Detector(model_dir="dummy")

def test_detector_load_onnx_failure():
    with patch("pipeline.detect.Path.exists", return_value=True):
        with patch("onnxruntime.InferenceSession", side_effect=Exception("mock fail")):
            with patch("ultralytics.YOLO") as MockYOLO:
                d = Detector(model_dir="dummy")
                assert d.yolo_model is not None
                MockYOLO.assert_called_once()

def test_detect_empty_frame(empty_detector):
    high, low = empty_detector.detect(None)
    assert high == []
    assert low == []


def test_split_detections(empty_detector):
    preds = np.array([
        [320, 320, 100, 100, 0.9],   # High conf
        [100, 100, 50, 50, 0.4],     # Low conf
        [200, 200, 50, 50, 0.1],     # Ignore
    ])
    high, low = empty_detector._split_detections(preds, sx=1.0, sy=1.0)
    assert len(high) == 1
    assert high[0].confidence == 0.9
    assert len(low) == 1
    assert low[0].confidence == 0.4


def test_detect_onnx(empty_detector):
    # Mock ONNX session
    mock_session = MagicMock()
    mock_session.get_inputs.return_value[0].name = "input"
    
    # Return shape (1, 5, 1) mimicking transposed anchor preds
    mock_out = np.array([[[320], [320], [100], [100], [0.9]]])
    mock_session.run.return_value = [mock_out]
    
    empty_detector.onnx_session = mock_session
    
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    with patch("cv2.cvtColor", return_value=np.zeros((640, 640, 3))), \
         patch("cv2.resize", return_value=np.zeros((640, 640, 3))):
        high, low = empty_detector.detect(frame)
        assert len(high) == 1
        assert len(low) == 0
        assert mock_session.run.called


def test_detect_ultralytics(empty_detector):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_box.xyxy = np.array([[100.0, 100.0, 200.0, 200.0]])
    
    mock_box_ignore1 = MagicMock()
    mock_box_ignore1.conf = [0.9]
    mock_box_ignore1.cls = [1]  # Not a person
    
    mock_box_ignore2 = MagicMock()
    mock_box_ignore2.conf = [0.1]
    mock_box_ignore2.cls = [0]  # Low conf
    
    mock_result.boxes = [mock_box, mock_box_ignore1, mock_box_ignore2]
    
    empty_detector.yolo_model.return_value = [mock_result]
    
    high, low = empty_detector.detect(frame)
    assert len(high) == 1
    assert high[0].confidence == 0.9
    assert len(low) == 0
