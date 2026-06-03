-- 002_remaining_tables.sql
-- Applied manually when 001_init.sql only partially ran due to cached Docker image.
-- On a fresh docker compose down -v && docker compose up, 001_init.sql runs completely.

CREATE TABLE IF NOT EXISTS zone_events (
    event_id UUID NOT NULL DEFAULT gen_random_uuid(),
    track_id INT NOT NULL,
    id_token TEXT,
    store_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    zone_name TEXT NOT NULL,
    zone_type TEXT NOT NULL,
    is_revenue_zone BOOLEAN NOT NULL DEFAULT true,
    event_type TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    dwell_ms INT,
    zone_hotspot_x FLOAT,
    zone_hotspot_y FLOAT,
    gender TEXT,
    age INT,
    age_bucket TEXT,
    is_staff BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (event_id, event_time)
);
SELECT create_hypertable('zone_events', 'event_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_zone_events_store_zone_ts ON zone_events (store_id, zone_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_zone_events_id_token ON zone_events (id_token, event_time DESC);

CREATE TABLE IF NOT EXISTS queue_events (
    queue_event_id UUID NOT NULL DEFAULT gen_random_uuid(),
    track_id INT NOT NULL,
    id_token TEXT,
    store_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    zone_name TEXT,
    event_type TEXT NOT NULL,
    queue_join_ts TIMESTAMPTZ NOT NULL,
    queue_served_ts TIMESTAMPTZ,
    queue_exit_ts TIMESTAMPTZ NOT NULL,
    wait_seconds INT NOT NULL,
    queue_position_at_join INT NOT NULL,
    abandoned BOOLEAN NOT NULL DEFAULT false,
    zone_hotspot_x FLOAT,
    zone_hotspot_y FLOAT,
    gender TEXT,
    age INT,
    age_bucket TEXT,
    PRIMARY KEY (queue_event_id, queue_join_ts)
);
SELECT create_hypertable('queue_events', 'queue_join_ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_queue_events_store_ts ON queue_events (store_id, queue_join_ts DESC);
CREATE INDEX IF NOT EXISTS idx_queue_events_id_token ON queue_events (id_token, queue_join_ts DESC);

CREATE TABLE IF NOT EXISTS pos_transactions (
    order_id INT NOT NULL PRIMARY KEY,
    order_date DATE NOT NULL,
    order_time TIME NOT NULL,
    store_id TEXT NOT NULL,
    product_id INT NOT NULL,
    brand_name TEXT,
    total_amount NUMERIC(12,2) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pos_store_date_time ON pos_transactions (store_id, order_date, order_time);

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_visitor_metrics
WITH (timescaledb.continuous) AS
SELECT store_id, time_bucket('1 hour', event_timestamp) AS bucket,
    COUNT(DISTINCT id_token) FILTER (WHERE event_type = 'entry' AND is_staff = false) AS unique_visitors
FROM entry_exit_events GROUP BY store_id, bucket;

SELECT add_continuous_aggregate_policy('hourly_visitor_metrics',
    start_offset => INTERVAL '1 month', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_zone_dwell
WITH (timescaledb.continuous) AS
SELECT store_id, zone_id, zone_name, time_bucket('1 hour', event_time) AS bucket,
    COUNT(DISTINCT id_token) FILTER (WHERE event_type = 'zone_entered') AS visit_count,
    AVG(dwell_ms) FILTER (WHERE event_type IN ('zone_exited', 'zone_dwell') AND is_staff = false) AS avg_dwell_ms
FROM zone_events GROUP BY store_id, zone_id, zone_name, bucket;

SELECT add_continuous_aggregate_policy('hourly_zone_dwell',
    start_offset => INTERVAL '1 month', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE);

CREATE OR REPLACE VIEW visitor_sessions AS
SELECT e.id_token, e.store_id, e.event_type AS entry_event_type,
    e.event_timestamp AS entry_ts, e.is_staff, e.gender_pred, e.age_bucket
FROM entry_exit_events e WHERE e.event_type IN ('entry', 'reentry');
