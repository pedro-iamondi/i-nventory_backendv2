from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os
from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded
import time

app = Flask(__name__)
CORS(app)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-vision-key.json"
client = vision.ImageAnnotatorClient()

last_request_time = 0

@app.route("/api/count", methods=["POST"])
def count_objects():
    global last_request_time

    # Proteção contra spam (máx 1 requisição por segundo)
    current_time = time.time()
    if current_time - last_request_time < 1.2:
        return jsonify({
            "count": 0,
            "annotated_image": data.get("image", ""),
            "message": "Aguarde 1 segundo entre contagens"
        }), 429

    last_request_time = current_time

    data = request.get_json()
    img_b64 = data.get("image")
    user_input = data.get("item_name", "").strip().lower()

    if not img_b64:
        return jsonify({"error": "no image"}), 400

    try:
        _, encoded = img_b64.split(",", 1)
    except:
        encoded = img_b64

    content = base64.b64decode(encoded)
    image = vision.Image(content=content)

    try:
        response = client.object_localization(image=image, timeout=10)
    except (ServiceUnavailable, DeadlineExceeded):
        return jsonify({
            "count": "timeout",
            "annotated_image": img_b64,
            "message": "Google Vision temporariamente indisponível. Tente novamente em 2 segundos."
        })

    objects = response.localized_object_annotations

    # Filtro inteligente por tipo
    detected = []
    for obj in objects:
        if obj.score < 0.7:
            continue
        box = obj.bounding_poly.normalized_vertices
        w = box[2].x - box[0].x
        h = box[2].y - box[0].y
        aspect = w / h if h > 0 else 1

        if "bateria" in user_input or "cr2477" in user_input:
            if 0.7 <= aspect <= 1.4 and 0.02 <= w <= 0.16:
                detected.append(obj)
        elif "caneta" in user_input:
            if aspect < 0.6 and w > 0.08:
                detected.append(obj)
        elif "parafuso" in user_input:
            if aspect < 0.5 and w < 0.12:
                detected.append(obj)
        else:
            if 0.015 <= w <= 0.22:
                detected.append(obj)

    count = len(detected)

    # Desenha caixinhas
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    img_show = img.copy()

    for i, obj in enumerate(detected, 1):
        box = obj.bounding_poly.normalized_vertices
        x1 = int(box[0].x * img.shape[1])
        y1 = int(box[0].y * img.shape[0])
        x2 = int(box[2].x * img.shape[1])
        y2 = int(box[2].y * img.shape[0])
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 12)
        cv2.putText(img_show, str(i), (x1 + 25, y1 + 85),
                    cv2.FONT_HERSHEY_DUPLEX, 4.5, (0, 255, 0), 12)

    _, buffer = cv2.imencode(".jpg", img_show)
    img_out = base64.b64encode(buffer).decode()

    item_name = user_input.title() if user_input else "objeto(s)"
    return jsonify({
        "count": count,
        "annotated_image": f"data:image/jpeg;base64,{img_out}",
        "message": f"Detectados {count} × {item_name}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)