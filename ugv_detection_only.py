import cv2
from ultralytics import YOLO
# Load YOLO model
model = YOLO("yolo26n.onnx")
# Open USB camera
cap = cv2.VideoCapture(0)
# Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run YOLO detection
    results = model(frame)
    annotated_frame = frame.copy()
    # Get class names
    names = model.names
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{names[cls]} {conf:.2f}"
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            # Draw label
            cv2.putText(annotated_frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)
    # Calculate FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time if inference_time > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    # Show output window
    cv2.imshow("YOLO Detection", annotated_frame)
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
