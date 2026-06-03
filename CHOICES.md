# Architectural Choices

## Choice 1: Three Separate Event Schemas (Not Unified)

**Decision**: The event ingestion endpoint (`POST /events/ingest`) accepts **three distinct JSON schemas** (entry/exit, zone, queue) rather than a unified 11-field schema.

**Why**: The `sample_events.jsonl` ground truth and the DB schema in `design.md` both use 3 separate tables with different fields. Implementing a unified schema would require lossy projection of all event types onto a single record shape, losing critical fields like `zone_hotspot_x/y`, `queue_position_at_join`, and `dwell_ms`. The three-schema approach preserves all semantically meaningful data and enables richer analytics (e.g., direct `JOIN queue_events ON pos_transactions` for the funnel endpoint without needing to unpack a metadata JSONB blob).

**Trade-off**: The ingest endpoint must fingerprint the event type from the payload (using field presence — e.g., `queue_event_id` → queue event). This adds a thin layer of routing logic but removes the need for a mandatory `event_type_discriminator` field in every payload.

---

## Choice 2: Physical mm Coordinate System for Zone Definitions

**Decision**: Zone polygons in `store_layout/zones.json` are defined in **physical floor millimetres** (matching layout PNG architectural dimensions), not in camera pixel coordinates.

**Why**: The layout PNGs include explicit architectural measurements (e.g., 2594mm, 1347mm, 2000mm, 4020mm). Defining zones in physical mm and using per-camera homography matrices to project pixel foot-points onto this physical ground plane achieves two critical goals: (1) eliminates perspective distortion for accurate ZONE_DWELL attribution, and (2) enables cross-camera deduplication by comparing ground-plane positions across overlapping cameras (CAM1, CAM2, CAM3 in Store 1 all have overlapping FOVs).

**Trade-off**: Requires calibration (4 point correspondences per camera to compute H matrix). Initial calibration points are approximate and may need refinement after reviewing actual footage.

---

## Choice 3: Config-Supplied UTC Timestamps for Clip Synchronisation

**Decision**: Each clip's base UTC start timestamp is stored in `store_layout/camera_config.json` as a **config-supplied value** aligned to the POS transaction dates (Store 1 = `2026-03-08`, Store 2 = `2026-04-10`).

**Why**: The primary requirement for the `/funnel` endpoint is correlating `queue_exit_ts` (when a visitor left the billing zone) with POS transactions (when a purchase occurred). If the clip's base timestamp is derived from file modification time or a random UTC reference, the `queue_exit_ts` values will not fall within the 5-minute window of actual POS transaction timestamps, making conversion attribution impossible. By anchoring the base timestamp to the date visible in the sample data and POS data, all derived event timestamps are guaranteed to be temporally correlated with the transactions file, enabling accurate funnel calculations.

**Trade-off**: The exact time-of-day offset within the date (e.g., 12:00:00 UTC) is approximate. After reviewing the footage, the hour should be adjusted to match the first visible POS transaction of the day.
