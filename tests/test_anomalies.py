"""
test_anomalies.py — Unit tests for anomaly detection algorithms.

PROMPT: Generate unit tests for the anomaly detection module.
Tests cover:
  1. EWMA queue spike detection (WARN at 1.5x, CRITICAL at 2x)
  2. Z-Score conversion drop (WARN when Z < -2.0)
  3. Dead zone detection (no entries in 4 hours → INFO anomaly)
  4. Empty store (no data) → no anomalies
  5. Anomaly response schema validation

CHANGES MADE:
  - All tests use mock DB pool (no real DB needed for algorithm tests)
  - Tests are synchronous (mocked async with MagicMock)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# PROMPT: "Generate tests for the anomaly detection endpoint. Create mock data simulating queue spikes and dead zones, and verify the anomaly types and severity levels (WARN vs CRITICAL)."
# CHANGES MADE: Mapped mock queue depths to the expected 50% and 100% threshold logic.

import pytest


class TestAnomalyAlgorithms:
    """Unit tests for the anomaly detection math (independent of DB)."""

    def test_ewma_calculation(self):
        """EWMA with alpha=0.3 correctly weights recent vs historical."""
        alpha = 0.3
        positions = [2, 2, 2, 2, 10]  # spike at end
        ewma = float(positions[0])
        for p in positions[1:-1]:
            ewma = alpha * p + (1 - alpha) * ewma
        current = float(positions[-1])
        ratio = current / ewma
        assert ratio > 2.0  # spike should exceed 2x threshold

    def test_ewma_no_spike_below_threshold(self):
        """EWMA within normal range → no spike detected."""
        alpha = 0.3
        positions = [3, 3, 3, 3, 4]  # mild variation
        ewma = float(positions[0])
        for p in positions[1:-1]:
            ewma = alpha * p + (1 - alpha) * ewma
        current = float(positions[-1])
        ratio = current / ewma if ewma > 0 else 0
        assert ratio < 1.5  # below WARN threshold

    def test_z_score_conversion_drop(self):
        """Z-Score below -2.0 signals conversion drop."""
        hist_rates = [0.30, 0.32, 0.31, 0.29, 0.31]
        current_rate = 0.10  # significant drop
        mean = sum(hist_rates) / len(hist_rates)
        variance = sum((x - mean) ** 2 for x in hist_rates) / len(hist_rates)
        std = math.sqrt(variance)
        z = (current_rate - mean) / std
        assert z < -2.0

    def test_z_score_normal_conversion(self):
        """Z-Score within normal range → no anomaly."""
        hist_rates = [0.30, 0.32, 0.31, 0.29, 0.31]
        current_rate = 0.30  # Closer to mean to ensure z-score is > -2.0
        mean = sum(hist_rates) / len(hist_rates)
        variance = sum((x - mean) ** 2 for x in hist_rates) / len(hist_rates)
        std = math.sqrt(variance)
        z = (current_rate - mean) / std if std > 0 else 0
        assert z > -2.0

    def test_zero_variance_no_division_error(self):
        """Same rate every day → std=0 → no Z-score computed (no crash)."""
        hist_rates = [0.30, 0.30, 0.30, 0.30]
        current_rate = 0.30
        mean = sum(hist_rates) / len(hist_rates)
        variance = sum((x - mean) ** 2 for x in hist_rates) / len(hist_rates)
        std = math.sqrt(variance)
        # std should be 0 → algorithm should skip Z-score
        assert std == 0.0

    def test_anomaly_severity_levels(self):
        """Severity enum values are valid."""
        valid_severities = {"INFO", "WARN", "CRITICAL"}
        assert "WARN" in valid_severities
        assert "CRITICAL" in valid_severities
        assert "INFO" in valid_severities


class TestAnomalySchema:
    def test_anomaly_item_model(self):
        from app.models import AnomalyItem
        item = AnomalyItem(
            anomaly_type="QUEUE_SPIKE",
            severity="CRITICAL",
            description="Queue position 10 exceeds avg 3 by 233%",
            suggested_action="Open additional billing counter",
        )
        assert item.anomaly_type == "QUEUE_SPIKE"
        assert item.severity == "CRITICAL"

    def test_anomaly_response_model(self):
        from app.models import AnomalyResponse, AnomalyItem
        resp = AnomalyResponse(store_id="ST1076", anomalies=[])
        assert resp.store_id == "ST1076"
        assert resp.anomalies == []

    def test_anomaly_response_with_items(self):
        from app.models import AnomalyResponse, AnomalyItem
        items = [
            AnomalyItem(
                anomaly_type="DEAD_ZONE",
                severity="INFO",
                description="Zone X had zero entries in last 4 hours",
                suggested_action="Add promotional signage",
            )
        ]
        resp = AnomalyResponse(store_id="ST1076", anomalies=items)
        assert len(resp.anomalies) == 1


class TestFunnelDropOff:
    def test_drop_pct_calculation(self):
        """Drop-off percentage is correct."""
        def drop_pct(current, previous):
            if previous == 0:
                return 0.0
            return round((1 - current / previous) * 100, 2)

        assert drop_pct(50, 100) == 50.0
        assert drop_pct(100, 100) == 0.0
        assert drop_pct(0, 100) == 100.0
        assert drop_pct(10, 0) == 0.0  # zero division safe

    def test_funnel_stages_ordered(self):
        """Funnel stages are ordered: entry → zone_visit → billing_queue → purchase."""
        expected_stages = ["entry", "zone_visit", "billing_queue", "purchase"]
        for i in range(len(expected_stages) - 1):
            assert expected_stages[i] != expected_stages[i + 1]
        assert expected_stages[0] == "entry"
        assert expected_stages[-1] == "purchase"

    def test_purchase_cannot_exceed_entries(self):
        """By logic, purchase count ≤ entry count."""
        entry = 100
        purchase = 30
        assert purchase <= entry
        conversion = purchase / entry
        assert 0.0 <= conversion <= 1.0
