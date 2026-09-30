from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

target_classes = ["person", "backpack", "handbag", "suitcase"]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            name = model.names[cls]

            if name in target_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0,255,0) if name=="person" else (0,0,255)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,name,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,color,2)

    cv2.imshow("AI Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()