import cv2, socket, json, threading, time, base64
from ultralytics import YOLO

MODEL_PATH = "yolo26n.onnx"
HOST       = "0.0.0.0"
PORT       = 9000

model = YOLO(MODEL_PATH, task="detect")
print("Model loaded.")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

import subprocess
ip = subprocess.getoutput("hostname -I").strip().split()[0]
print(f"Pi IP: {ip}  |  Port: {PORT}")
print("Waiting for laptop...")

conn, addr = server.accept()
print(f"Laptop connected from {addr}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Streaming...\n")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    results    = model(frame, verbose=False, conf=0.25)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "label": results[0].names[int(box.cls)],
            "conf":  round(float(box.conf), 2),
            "box":   [round(x, 1) for x in box.xyxy[0].tolist()]
        })

    inf_ms = results[0].speed['inference']
    fps    = 1000 / inf_ms if inf_ms > 0 else 0

    # Encode frame as JPEG (quality 50 keeps it lightweight)
    _, jpeg    = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 50])
    frame_b64  = base64.b64encode(jpeg.tobytes()).decode()

    payload = json.dumps({
        "frame":      frame_count,
        "fps":        round(fps, 1),
        "detections": detections,
        "img":        frame_b64
    }) + "\n"

    try:
        conn.sendall(payload.encode())
    except BrokenPipeError:
        print("Laptop disconnected.")
        break

cap.release()
conn.close()
server.close()
