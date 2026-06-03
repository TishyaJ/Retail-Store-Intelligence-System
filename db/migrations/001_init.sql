-- ============================================================
-- 001_init.sql — Retail Store Intelligence System
-- TimescaleDB schema: 3 hypertables + pos_transactions + aggregates
-- ============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================
-- TABLE 1: entry_exit_events
-- Entry/exit/reentry events from CAM3/entry cameras
-- ============================================================
CREATE TABLE IF NOT EXISTS entry_exit_events (
    event_id          UUID          NOT NULL DEFAULT gen_random_uuid(),
    id_token          TEXT          NOT NULL,        -- e.g. ID_60001
    store_code        TEXT,                          -- raw: e.g. store_1076
    store_id          TEXT          NOT NULL,        -- normalised: e.g. ST1076
    camera_id         TEXT          NOT NULL,
    event_type        TEXT          NOT NULL,        -- entry / exit / reentry
    event_timestamp   TIMESTAMPTZ   NOT NULL,
    is_staff          BOOLEAN       NOT NULL DEFAULT false,
    gender_pred       TEXT,                          -- M / F / Unknown
    age_pred          INT,
    age_bucket        TEXT,                          -- e.g. 25-34
    is_face_hidden    BOOLEAN       DEFAULT false,
    group_id          TEXT,                          -- e.g. G_10 (null if solo)
    group_size        INT,                           -- null if solo
    confidence        FLOAT         NOT NULL DEFAULT 1.0,
    -- TimescaleDB requires partition column (event_timestamp) in PK
    PRIMARY KEY (event_id, event_timestamp)
);

SELECT create_hypertable(
    'entry_exit_events', 'event_timestamp',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_entry_exit_store_ts
    ON entry_exit_events (store_id, event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_entry_exit_id_token
    ON entry_exit_events (id_token, event_timestamp DESC);

-- ============================================================
-- TABLE 2: zone_events
-- Zone entered/exited/dwell events from zone cameras (CAM1/CAM2/CAM_ZONE)
-- ============================================================
CREATE TABLE IF NOT EXISTS zone_events (
    event_id          UUID          NOT NULL DEFAULT gen_random_uuid(),
    track_id          INT           NOT NULL,        -- per-camera tracker ID
    id_token          TEXT,                          -- linked to entry_exit_events
    store_id          TEXT          NOT NULL,        -- e.g. ST1076
    camera_id         TEXT          NOT NULL,
    zone_id           TEXT          NOT NULL,        -- e.g. PURPLLE_MUM_1076_Z01
    zone_name         TEXT          NOT NULL,
    zone_type         TEXT          NOT NULL,        -- SHELF / DISPLAY / BILLING
    is_revenue_zone   BOOLEAN       NOT NULL DEFAULT true,
    event_type        TEXT          NOT NULL,        -- zone_entered / zone_exited / zone_dwell
    event_time        TIMESTAMPTZ   NOT NULL,
    dwell_ms          INT,                           -- null for zone_entered; ms since enter for others
    zone_hotspot_x    FLOAT,
    zone_hotspot_y    FLOAT,
    gender            TEXT,
    age               INT,
    age_bucket        TEXT,
    is_staff          BOOLEAN       NOT NULL DEFAULT false,
    -- TimescaleDB requires partition column (event_time) in PK
    PRIMARY KEY (event_id, event_time)
);

SELECT create_hypertable(
    'zone_events', 'event_time',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_zone_events_store_zone_ts
    ON zone_events (store_id, zone_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_zone_events_id_token
    ON zone_events (id_token, event_time DESC);

-- ============================================================
-- TABLE 3: queue_events
-- Billing queue completed/abandoned events from CAM5/CAM_BILLING
-- ============================================================
CREATE TABLE IF NOT EXISTS queue_events (
    queue_event_id        UUID          NOT NULL DEFAULT gen_random_uuid(),
    track_id              INT           NOT NULL,
    id_token              TEXT,
    store_id              TEXT          NOT NULL,
    camera_id             TEXT          NOT NULL,
    zone_id               TEXT          NOT NULL,   -- e.g. PURPLLE_MUM_1076_Z_BILLING_01
    zone_name             TEXT,
    event_type            TEXT          NOT NULL,   -- queue_completed / queue_abandoned
    queue_join_ts         TIMESTAMPTZ   NOT NULL,
    queue_served_ts       TIMESTAMPTZ,              -- null if abandoned
    queue_exit_ts         TIMESTAMPTZ   NOT NULL,
    wait_seconds          INT           NOT NULL,
    queue_position_at_join INT          NOT NULL,
    abandoned             BOOLEAN       NOT NULL DEFAULT false,
    zone_hotspot_x        FLOAT,
    zone_hotspot_y        FLOAT,
    gender                TEXT,
    age                   INT,
    age_bucket            TEXT,
    -- TimescaleDB requires partition column (queue_join_ts) in PK
    PRIMARY KEY (queue_event_id, queue_join_ts)
);

SELECT create_hypertable(
    'queue_events', 'queue_join_ts',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_queue_events_store_ts
    ON queue_events (store_id, queue_join_ts DESC);

CREATE INDEX IF NOT EXISTS idx_queue_events_id_token
    ON queue_events (id_token, queue_join_ts DESC);

-- ============================================================
-- TABLE 4: pos_transactions
-- POS data from POS_transactions.csv
-- CSV schema: order_id, order_date (DD-MM-YYYY), order_time (HH:MM:SS),
--             store_id, product_id, brand_name, total_amount
-- Multiple rows per (store_id, order_date, order_time) = one basket (line items)
-- ============================================================
CREATE TABLE IF NOT EXISTS pos_transactions (
    order_id        INT           NOT NULL,
    order_date      DATE          NOT NULL,          -- parsed from DD-MM-YYYY string
    order_time      TIME          NOT NULL,          -- parsed from HH:MM:SS string
    store_id        TEXT          NOT NULL,          -- e.g. ST1008
    product_id      INT           NOT NULL,
    brand_name      TEXT,
    total_amount    NUMERIC(12,2) NOT NULL,
    PRIMARY KEY (order_id)
);

CREATE INDEX IF NOT EXISTS idx_pos_store_date_time
    ON pos_transactions (store_id, order_date, order_time);

-- ============================================================
-- CONTINUOUS AGGREGATES
-- ============================================================

-- Hourly unique visitor counts (powers /metrics endpoint)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_visitor_metrics
WITH (timescaledb.continuous) AS
SELECT
    store_id,
    time_bucket('1 hour', event_timestamp) AS bucket,
    COUNT(DISTINCT id_token) FILTER (WHERE event_type = 'entry' AND is_staff = false) AS unique_visitors
FROM entry_exit_events
GROUP BY store_id, bucket;

SELECT add_continuous_aggregate_policy(
    'hourly_visitor_metrics',
    start_offset      => INTERVAL '1 month',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => TRUE
);

-- Hourly zone dwell averages (powers /heatmap endpoint)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_zone_dwell
WITH (timescaledb.continuous) AS
SELECT
    store_id,
    zone_id,
    zone_name,
    time_bucket('1 hour', event_time) AS bucket,
    COUNT(DISTINCT id_token) FILTER (WHERE event_type = 'zone_entered') AS visit_count,
    AVG(dwell_ms) FILTER (WHERE event_type IN ('zone_exited', 'zone_dwell') AND is_staff = false) AS avg_dwell_ms
FROM zone_events
GROUP BY store_id, zone_id, zone_name, bucket;

SELECT add_continuous_aggregate_policy(
    'hourly_zone_dwell',
    start_offset      => INTERVAL '1 month',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => TRUE
);

-- ============================================================
-- HELPER VIEW: visitor_sessions
-- Links id_token (from entry/exit) with track_id (zone/queue)
-- via store_id + time proximity for funnel computation
-- ============================================================
CREATE OR REPLACE VIEW visitor_sessions AS
SELECT
    e.id_token,
    e.store_id,
    e.event_type       AS entry_event_type,
    e.event_timestamp  AS entry_ts,
    e.is_staff,
    e.gender_pred,
    e.age_bucket
FROM entry_exit_events e
WHERE e.event_type IN ('entry', 'reentry');
