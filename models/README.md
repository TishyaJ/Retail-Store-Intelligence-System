# Models Directory

Place trained model weight files here before running the vision pipeline.

## Required Model Files

| File | Size (approx) | Source |
|------|--------------|--------|
| `yolov11m_retail.onnx` | ~50 MB | See Colab notebook: `colab/train_yolov11m.ipynb` |
| `mobilenet_staff.onnx` | ~9 MB | See Colab notebook: `colab/train_mobilenet_staff.ipynb` |
| `osnet_x0_25.pth` | ~2 MB | Auto-downloaded by torchreid on first pipeline run |

## Current Status

- **YOLOv11m**: Using pretrained weights via `ultralytics` (no retail fine-tuning yet). Drop in `yolov11m_retail.onnx` when Colab training is complete.
- **MobileNetV3 Staff Classifier**: Will use HSV-only mode until `mobilenet_staff.onnx` is available from Colab training.
- **OSNet Re-ID**: Auto-downloaded via `torchreid` model zoo on first run. No action needed.

## Swapping in Custom Weights

1. Export your trained models from Colab to ONNX format
2. Download the `.onnx` files from your Google Drive
3. Place them in this `models/` directory with the exact filenames above
4. Restart the `vision-pipeline` container: `docker compose restart vision-pipeline`
5. No code changes required — the pipeline checks for ONNX files at startup

## Training Notebooks

- YOLOv11m detection: [`colab/train_yolov11m.ipynb`](../colab/train_yolov11m.ipynb)
- MobileNetV3 staff classifier: [`colab/train_mobilenet_staff.ipynb`](../colab/train_mobilenet_staff.ipynb)
