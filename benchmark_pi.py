import time, psutil, os
from ultralytics import YOLO

BASE     = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(BASE, "test_images")
MODELS   = {
    "YOLOv8n": os.path.join(BASE, "yolov8n.onnx"),
    "YOLO11n": os.path.join(BASE, "yolo11n.onnx"),
    "YOLO26n": os.path.join(BASE, "yolo26n.onnx"),
}

images = [
    os.path.join(IMG_DIR, f)
    for f in os.listdir(IMG_DIR)
    if f.endswith(('.jpg', '.png'))
][:100]

print(f"Running benchmark on {len(images)} images\n")
print(f"{'Model':<12} {'Avg(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'FPS':<10} {'RAM(MB)'}")
print("-" * 65)

for name, path in MODELS.items():
    model = YOLO(path, task="detect")

    # Warmup — don't measure this
    model(images[0], verbose=False)
    model(images[0], verbose=False)

    latencies = []
    ram_before = psutil.virtual_memory().used / 1024**2

    for img in images:
        t0 = time.perf_counter()
        model(img, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    ram_after  = psutil.virtual_memory().used / 1024**2
    avg_ms     = sum(latencies) / len(latencies)
    min_ms     = min(latencies)
    max_ms     = max(latencies)
    fps        = 1000 / avg_ms
    ram_used   = ram_after - ram_before

    print(f"{name:<12} {avg_ms:<12.1f} {min_ms:<12.1f} {max_ms:<12.1f} {fps:<10.2f} {ram_used:<.1f}")