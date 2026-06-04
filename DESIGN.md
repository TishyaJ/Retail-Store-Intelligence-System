# Retail Store Intelligence System — Technical Design

This document details the architectural design and end-to-end data flow for the Purplle Retail Store Intelligence System.

---

## 1. End-to-End Data Flow

The system processes raw CCTV footage into actionable business intelligence through a strict decoupling of vision processing (ML) and data serving (API).

### Phase A: Vision Pipeline (Per-Camera Processing)
1. **Video Ingestion:** `cv2.VideoCapture` pulls frames sequentially.
2. **Object Detection:** Frames are passed to **YOLOv11m** (exported via ONNX for CPU inference). The model yields bounding boxes and confidence scores for the `person` class.
3. **Tracking:** **ByteTrack** associates frame-by-frame detections into coherent tracklets using Kalman filtering and spatial IoU matching.
4. **Re-Identification (ReID):** Crops of tracked individuals are passed through **OSNet** to extract a 512-dimensional visual embedding. 
5. **Staff Classification:** **MobileNetV3-Small** evaluates the crop against the staff uniform binary classifier. A temporal voting mechanism assigns a final `is_staff` boolean to the track.
6. **Cross-Camera Deduplication:** The pixel coordinate of the person's feet is projected onto a unified 2D ground plane using a **Homography Matrix** (`H`). If the ReID embedding matches a known person close by on the ground plane, the local `track_id` is merged into a global `id_token`.
7. **Zone Engine:** The ground coordinates are hit-tested against defined Shapely polygons (Entry, Zone, Billing). The engine applies **temporal hysteresis** (e.g., 2000ms dwell required) to filter boundary noise and manages the queue state machine.
8. **Event Emission:** The Zone Engine yields discrete JSON events (`ENTRY`, `ZONE_DWELL`, etc.). The Event Emitter buffers these and flushes them to the API Server via HTTP POST.

### Phase B: Data Ingestion & Storage
1. **API Ingestion:** The `api-server` receives batches at `POST /events/ingest`.
2. **Schema Validation:** FastAPI/Pydantic strictly validates the incoming payloads.
3. **Persistence:** Valid events are inserted into **TimescaleDB** (PostgreSQL extension). The `events` tables are partitioned by timestamp.
4. **Aggregation:** TimescaleDB Continuous Aggregates incrementally pre-calculate hourly and daily rollups of visitor counts, average dwells, and queue depths in the background.

### Phase C: Analytics & Serving
1. **Client Request:** An analyst queries `GET /stores/ST1076/metrics`.
2. **Query Execution:** The `api-server` runs fast queries against the continuous aggregates and real-time partitions.
3. **POS Correlation:** If the funnel endpoint is hit, the `POS_Correlator` maps raw `queue_exit` events against loaded `POS_transactions.csv` timestamps using a 5-minute lookback window to attribute sales.
4. **Response:** Metrics are serialized to JSON and returned to the client in <200ms.

---

## 2. AI-Assisted Decisions

This project heavily leveraged AI (Antigravity/Claude/Gemini) for architectural planning and technical implementation. Below are three critical design choices where AI assistance drove the outcome:

### Decision 1: Model Selection & ONNX Strategy
* **Prompt used:** *"For the vision pipeline, should we fine-tune YOLOv8 or use YOLOv11m off the shelf? Also, considering we are running purely on CPU for the submission, what is the best format?"*
* **Outcome Adopted:** We adopted YOLOv11m pre-trained on COCO but stripped down to a 1-class (person) model. AI recommended exporting to ONNX format with ONNXRuntime, explaining that PyTorch overhead on CPU is significant, and ONNX graph optimizations would provide a ~30% inference speedup without requiring GPU access.

### Decision 2: Cross-Camera Deduplication Architecture
* **Prompt used:** *"How do we prevent duplicate visitor counts when the same person walks from CAM1 to CAM3? Store 1 has overlapping cameras."*
* **Outcome Adopted:** The AI recommended a hybrid spatial-visual approach. Visual embeddings (OSNet) alone are prone to false positives in uniform environments. By implementing a Homography Matrix to project pixel coordinates to a top-down ground plane, we constrain ReID matching to a tight physical radius (e.g., <2 meters), drastically reducing duplicate ID errors.

### Decision 3: Event Emission Decoupling
* **Prompt used:** *"Should the vision pipeline write directly to PostgreSQL or send events over an API?"*
* **Outcome Adopted:** We adopted an API ingestion model (`POST /events/ingest`). The AI correctly pointed out that allowing 8 concurrent ML processes to hold active database connections would lead to connection pool exhaustion and locking issues. By placing the API Server between the pipeline and the DB, we enforce Pydantic schema validation at the edge and allow the pipeline to batch-send events ephemerally.
