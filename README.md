# Indian Road Sign Detection — Edge Deployment on Raspberry Pi 5

## Overview
Comparative study of YOLOv8n, YOLO11n, and YOLO26n for Indian road sign 
detection, with edge deployment and benchmarking on Raspberry Pi 5.

Part of an ongoing IEEE paper submission.

---

## Dataset
- **Source:** Indian Traffic Sign Dataset (SDI, Roboflow)
- **License:** CC BY 4.0
- **Classes:** 85 Indian road sign categories (IRC standard)
- **splits:** 4690 train / 1383 val / test images
- **URL:** https://universe.roboflow.com/sdi/indian-traffic-sign/dataset/1

---

## Models
All three models fine-tuned on the Indian traffic sign dataset for 100 epochs.

| Model   | mAP@50 | mAP@50-95 | Precision | Recall | Params | ONNX size |
|---------|--------|-----------|-----------|--------|--------|-----------|
| YOLOv8n | 0.938  | 0.891     | 0.825     | 0.925  | 3.20M  | 12.4 MB   |
| YOLO11n | 0.953  | 0.915     | 0.858     | 0.951  | 2.62M  | 10.3 MB   |
| YOLO26n | 0.911  | 0.870     | 0.873     | 0.852  | 2.58M  | 9.5 MB    |

---

## Edge Deployment — Raspberry Pi 5

| Model   | Avg latency (ms) | FPS  | Min (ms) | Max (ms) |
|---------|-----------------|------|----------|----------|
| YOLOv8n | 163.8           | 6.11 | 161.5    | 168.6    |
| YOLO11n | 153.3           | 6.52 | 152.0    | 156.0    |
| YOLO26n | 134.4           | 7.44 | 132.8    | 136.0    |

**Key finding:** YOLO26n is 17.9% faster than YOLO11n and 22.0% faster 
than YOLOv8n on Pi 5 CPU, despite ranking lowest on GPU accuracy — 
attributed to its NMS-free detection head eliminating CPU-bound 
post-processing overhead.

---

## Training Setup
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
- **Framework:** Ultralytics 8.4.23
- **PyTorch:** 2.5.1+cu121
- **Epochs:** 100
- **Image size:** 640×640
- **Batch size:** 8
- **Optimizer:** AdamW (auto)
- **Patience:** 20 (early stopping)

---

## Deployment Setup
- **Hardware:** Raspberry Pi 5
- **Runtime:** ONNX Runtime 1.24.4 (CPUExecutionProvider)
- **Format:** ONNX (opset 19)
- **Benchmark images:** 50 test images

---

## Repository Structure
```
├── yolov8n.onnx              # Fine-tuned YOLOv8n
├── yolo11n.onnx              # Fine-tuned YOLO11n
├── yolo26n.onnx              # Fine-tuned YOLO26n
├── benchmark_pi.py           # Pi 5 benchmarking script
├── visualize_detections.py   # Detection visualization script
├── test_images/              # Sample test images
└── detection_results/        # Visual detection outputs per model
    ├── YOLOv8n/
    ├── YOLO11n/
    └── YOLO26n/
```

---

## How to Run on Raspberry Pi 5
```bash
# Clone repo
git clone https://github.com/YOURUSERNAME/ugv-indian-signs-models.git
cd ugv-indian-signs-models

# Set up virtual environment
python3 -m venv ~/bench_env
source ~/bench_env/bin/activate
pip install ultralytics onnxruntime onnx psutil

# Run benchmark
python benchmark_pi.py

# Run visual detection
python visualize_detections.py
```

---

## Citation
If you use this work, please cite:
> [Paper citation will be added after publication]

Dataset citation:
> SDI. Indian Traffic Sign Dataset. Roboflow Universe, 2023.
> https://universe.roboflow.com/sdi/indian-traffic-sign/dataset/1