from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8x.pt")

@app.route("/api/count", methods=["POST"])
def count():
    data = request.get_json()
    img_b64 = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    results = model(img, conf=0.1, iou=0.4, imgsz=1280)[0]
    boxes = results.boxes

    # Filtro final que resolve tudo nas suas fotos
    valid = []
    for box in boxes:
        w = box.xyxy[0][2] - box.xyxy[0][0]
        h = box.xyxy[0][3] - box.xyxy[0][1]
        if 40 <= w <= 280 and 40 <= h <= 280:
            valid.append(box)

    count = len(valid)
    img_show = img.copy()
    for i, box in enumerate(valid, 1):
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cv2.rectangle(img_show, (x1,y1), (x2,y2), (0,255,0), 8)
        cv2.putText(img_show, str(i), (x1+10, y1+60), cv2.FONT_HERSHEY_DUPLEX, 3, (0,255,0), 8)

    _, buf = cv2.imencode(".jpg", img_show)
    return jsonify({
        "count": count,
        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
        "message": f"Contados {count} itens (YOLO local)"
    })

if __name__ == "__main__":
    app.run(port=5000)