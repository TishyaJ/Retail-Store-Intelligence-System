#!/usr/bin/env bash
# run.sh — Multi-camera orchestration entry point
# Waits for API server, loads POS data, then launches all 8 camera processes.
set -euo pipefail

API_URL="${API_SERVER_URL:-http://api-server:8000}"
MAX_WAIT=60
RETRY_INTERVAL=5

echo "[run.sh] Waiting for API server at ${API_URL}/health ..."
elapsed=0
while true; do
    if curl -sf "${API_URL}/health" | grep -q '"status":"ok"'; then
        echo "[run.sh] API server is healthy."
        break
    fi
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "[run.sh] ERROR: API server not ready after ${MAX_WAIT}s. Exiting."
        exit 1
    fi
    sleep "$RETRY_INTERVAL"
    elapsed=$((elapsed + RETRY_INTERVAL))
done

echo "[run.sh] Loading POS transactions..."
curl -sf -X POST "${API_URL}/admin/pos/reload" || echo "[run.sh] WARNING: POS reload failed (non-fatal)"

echo "[run.sh] Starting vision pipeline for all cameras..."

# ---- Store 1 (ST1076) ----
# CAM1 and CAM2 overlap with CAM3 — cross-camera dedup is active
python -m pipeline.main_pipeline \
    --clip "Stores/Store 1/CAM 1 - zone.mp4" \
    --store_id ST1076 --camera_id CAM1 --camera_type zone &
PID_CAM1=$!

python -m pipeline.main_pipeline \
    --clip "Stores/Store 1/CAM 2 - zone.mp4" \
    --store_id ST1076 --camera_id CAM2 --camera_type zone &
PID_CAM2=$!

python -m pipeline.main_pipeline \
    --clip "Stores/Store 1/CAM 3 - entry.mp4" \
    --store_id ST1076 --camera_id CAM3 --camera_type entry &
PID_CAM3=$!

python -m pipeline.main_pipeline \
    --clip "Stores/Store 1/CAM 5 - billing.mp4" \
    --store_id ST1076 --camera_id CAM5 --camera_type billing &
PID_CAM5=$!

# ---- Store 2 (ST1008) ----
python -m pipeline.main_pipeline \
    --clip "Stores/Store 2/entry 1.mp4" \
    --store_id ST1008 --camera_id CAM_ENTRY_1 --camera_type entry &
PID_E1=$!

python -m pipeline.main_pipeline \
    --clip "Stores/Store 2/entry 2.mp4" \
    --store_id ST1008 --camera_id CAM_ENTRY_2 --camera_type entry &
PID_E2=$!

python -m pipeline.main_pipeline \
    --clip "Stores/Store 2/zone.mp4" \
    --store_id ST1008 --camera_id CAM_ZONE --camera_type zone &
PID_ZONE=$!

python -m pipeline.main_pipeline \
    --clip "Stores/Store 2/billing_area.mp4" \
    --store_id ST1008 --camera_id CAM_BILLING --camera_type billing &
PID_BILLING=$!

echo "[run.sh] All 8 camera processes launched."
echo "[run.sh] PIDs: CAM1=$PID_CAM1 CAM2=$PID_CAM2 CAM3=$PID_CAM3 CAM5=$PID_CAM5"
echo "[run.sh] PIDs: CAM_ENTRY_1=$PID_E1 CAM_ENTRY_2=$PID_E2 CAM_ZONE=$PID_ZONE CAM_BILLING=$PID_BILLING"

# Wait for all processes to finish
wait
echo "[run.sh] All camera processes completed."
