from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import httpx
import pytest

from pipeline.emit import EventEmitter, BATCH_MAX_SIZE, BUFFER_PATH


@pytest.fixture
def emitter():
    return EventEmitter("http://localhost:8000", "ST1076", "CAM1")


def test_push_buffers_events(emitter):
    emitter.push({"event_type": "test"})
    assert len(emitter._buffer) == 1
    # Should not flush yet
    assert getattr(emitter, "_post_with_retry", None) is not None


def test_push_flushes_at_max_size(emitter):
    with patch.object(emitter, "flush") as mock_flush:
        for _ in range(BATCH_MAX_SIZE - 1):
            emitter.push({"event": "test"})
        mock_flush.assert_not_called()
        emitter.push({"event": "test"})
        mock_flush.assert_called_once()


def test_flush_empty_does_nothing(emitter):
    with patch.object(emitter, "_post_with_retry") as mock_post:
        emitter.flush()
        mock_post.assert_not_called()


def test_flush_posts_events(emitter):
    emitter.push({"event": "test"})
    with patch("httpx.Client") as mock_client:
        mock_post = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"accepted": 1, "rejected": 0}
        mock_post.return_value = mock_response
        mock_client.return_value.__enter__.return_value.post = mock_post

        emitter.flush()

        mock_post.assert_called_once()
        assert len(emitter._buffer) == 0


def test_post_with_retry_writes_to_buffer_on_failure(emitter, tmp_path):
    emitter.push({"event": "test1"})
    emitter.push({"event": "test2"})
    
    # Force delays to 0 for fast testing
    with patch("pipeline.emit.BUFFER_PATH", str(tmp_path / "buffer.jsonl")):
        with patch("time.sleep"):  # skip sleeps
            with patch("httpx.Client") as mock_client:
                mock_post = MagicMock(side_effect=httpx.ConnectError("Connection failed"))
                mock_client.return_value.__enter__.return_value.post = mock_post
                
                emitter.flush()
                
                assert mock_post.call_count == 3  # 3 retries
                
                # Check buffer file
                buffer_file = tmp_path / "buffer.jsonl"
                assert buffer_file.exists()
                lines = buffer_file.read_text().strip().split("\n")
                assert len(lines) == 2
                assert json.loads(lines[0]) == {"event": "test1"}


def test_build_entry_exit_event(emitter):
    ts = datetime(2026, 3, 8, 10, 0, 0, tzinfo=timezone.utc)
    evt = emitter.build_entry_exit_event(
        id_token="ID_1",
        direction="entry",
        store_code="store_1",
        store_id="ST1076",
        camera_id="CAM1",
        ts=ts,
        is_staff=True,
    )
    assert evt["id_token"] == "ID_1"
    assert evt["event_type"] == "entry"
    assert evt["is_staff"] is True
    assert evt["event_timestamp"] == "2026-03-08T10:00:00.000000"
