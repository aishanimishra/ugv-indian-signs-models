import os
import cv2
from ultralytics import YOLO

BASE      = os.path.dirname(os.path.abspath(__file__))
IMG_DIR   = os.path.join(BASE, "test_images")
OUT_DIR   = os.path.join(BASE, "detection_results")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "YOLOv8n": os.path.join(BASE, "yolov8n.onnx"),
    "YOLO11n": os.path.join(BASE, "yolo11n.onnx"),
    "YOLO26n": os.path.join(BASE, "yolo26n.onnx"),
}

# Get test images
images = [
    os.path.join(IMG_DIR, f)
    for f in sorted(os.listdir(IMG_DIR))
    if f.endswith(('.jpg', '.png'))
][:10]  # first 10 images only

print(f"Running detection on {len(images)} images for each model\n")

for model_name, model_path in MODELS.items():
    print(f"--- {model_name} ---")
    model = YOLO(model_path, task="detect")

    # Create output folder per model
    model_out = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_out, exist_ok=True)

    for img_path in images:
        img_name = os.path.basename(img_path)
        results  = model(img_path, verbose=False, conf=0.25)[0]

        # Draw boxes on image
        annotated = results.plot()

        # Add model name label on top of image
        cv2.putText(
            annotated,
            f"Model: {model_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Save
        out_path = os.path.join(model_out, img_name)
        cv2.imwrite(out_path, annotated)
        print(f"  Saved: {out_path}")

    print(f"  Done — results in detection_results/{model_name}/\n")

print("All done. Results saved in detection_results/")
