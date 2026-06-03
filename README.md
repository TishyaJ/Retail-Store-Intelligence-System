# Retail Store Intelligence System
## Purplle Tech Challenge — PS3

**A containerised computer vision system that transforms raw CCTV footage into real-time retail analytics.**

---

## Quickstart — 5 Commands

```bash
# 1. Clone and configure
git clone <repo_url> && cd purplle
cp .env.example .env

# 2. Place video clips in Stores/ directory (provided by competition)
#    Store 1: Stores/Store 1/CAM 1 - zone.mp4, CAM 2 - zone.mp4, CAM 3 - entry.mp4, CAM 5 - billing.mp4
#    Store 2: Stores/Store 2/entry 1.mp4, entry 2.mp4, zone.mp4, billing_area.mp4

# 3. Build and start all services
docker compose up --build -d

# 4. Wait for the API to be healthy (auto-retried by vision pipeline)
curl http://localhost:8000/health

# 5. The vision pipeline starts automatically once API is healthy
#    To check events are flowing:
curl http://localhost:8000/stores/ST1076/metrics
```

---

## Architecture

```
CCTV Clips (8 cameras)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  vision-pipeline  (8 parallel processes)                 │
│  YOLOv11m → ByteTrack → OSNet ReID → ZoneEngine → Emit │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /events/ingest (batch 50)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  api-server  (FastAPI + asyncpg)                         │
│  /events/ingest │ /metrics │ /funnel │ /anomalies        │
└──────────────────────┬──────────────────────────────────┘
                       │ asyncpg
                       ▼
┌─────────────────────────────────────────────────────────┐
│  database  (TimescaleDB = PostgreSQL + time-series)      │
│  entry_exit_events │ zone_events │ queue_events │ POS    │
└─────────────────────────────────────────────────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/events/ingest` | Ingest batch (≤500) mixed events |
| `GET`  | `/health` | System health + feed freshness |
| `GET`  | `/stores/{id}/metrics` | Visitors, conversion rate, dwell, queue |
| `GET`  | `/stores/{id}/funnel` | 4-stage conversion funnel |
| `GET`  | `/stores/{id}/heatmap` | Zone visit frequency + dwell |
| `GET`  | `/stores/{id}/anomalies` | Queue spikes, conversion drops, dead zones |
| `POST` | `/admin/pos/reload` | Load POS_transactions.csv into DB |

## Event Schemas

Three separate event types (NOT a unified schema):

### Entry/Exit Events
```json
{"id_token":"ID_60001","store_code":"store_1076","camera_id":"cam1",
 "event_type":"entry","event_timestamp":"2026-03-08T10:00:00.000000",
 "is_staff":false,"gender_pred":"F","age_bucket":"25-34","confidence":0.91}
```

### Zone Events
```json
{"track_id":1,"store_id":"ST1076","camera_id":"CAM2",
 "zone_id":"PURPLLE_MUM_1076_Z01","zone_name":"Left Shelf","zone_type":"SHELF",
 "is_revenue_zone":"Yes","event_type":"zone_entered",
 "event_time":"2026-03-08T10:05:00.000000","dwell_ms":null}
```

### Queue Events
```json
{"queue_event_id":"uuid","track_id":200,"store_id":"ST1076","camera_id":"CAM5",
 "zone_id":"PURPLLE_MUM_1076_Z_BILLING_01","event_type":"queue_completed",
 "queue_join_ts":"2026-03-08T11:00:00.000000","queue_exit_ts":"2026-03-08T11:03:00.000000",
 "wait_seconds":180,"queue_position_at_join":2,"abandoned":false}
```

## Vision Pipeline

| Component | Technology | Notes |
|-----------|-----------|-------|
| Detection | YOLOv11m (ultralytics) | ONNX swappable; pretrained until Colab training completes |
| Tracking | ByteTrack | Dual-pass Hungarian matching; 30-frame LOST threshold |
| Re-ID | OSNet-x0_25 (torchreid) | Auto-downloaded; per-store gallery; 12-hour TTL |
| Staff | MobileNetV3 + HSV | ONNX swappable; HSV-only fallback |
| Dedup | Homography (cv2) | Physical mm ground plane; 500mm threshold |
| Zones | Shapely polygons | Physical mm coordinates; hysteresis inner/outer rings |

## Swapping Custom Models (Post-Training)

```bash
# Copy trained ONNX files to models/
cp /path/to/yolov11m_retail.onnx models/
cp /path/to/mobilenet_staff.onnx models/

# Restart pipeline — no code changes needed
docker compose restart vision-pipeline
```

## Running Tests

```bash
# Install test dependencies
pip install -r app/requirements.txt

# Run with coverage (70% threshold)
pytest tests/ -v --cov=app --cov=pipeline --cov-report=term-missing
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STORE1_CLIP_START_UTC` | `2026-03-08T12:00:00Z` | Store 1 clip base timestamp |
| `STORE2_CLIP_START_UTC` | `2026-04-10T12:00:00Z` | Store 2 clip base timestamp |
| `USE_ONNX_CPU` | `true` | Force CPU inference |
| `PIPELINE_BATCH_SIZE` | `50` | Events per HTTP batch |
| `LOG_LEVEL` | `INFO` | Structlog level |
