from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

@app.route("/api/count-simple", methods=["POST"])
def count_simple():
    data = request.get_json()
    img_b64 = data.get("image")

    if not img_b64:
        return jsonify({"error": "image missing"}), 400

    try:
        header, encoded = img_b64.split(",", 1)
    except ValueError:
        encoded = img_b64
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid image"}), 400

    # Pré-processamento melhorado
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)  # Aumente o blur para itens pequenos
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Inverte para objetos escuros

    # Operações morfológicas para separar objetos tocantes
    kernel = np.ones((5, 5), np.uint8)  # Ajuste kernel para itens finos como parafusos
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove ruído
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # Expande fundo

    # Usar Watershed para separar instâncias (melhor que contornos simples)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]  # Marca bordas (opcional, para debug)

    # Conta componentes únicos (exclui fundo)
    count = len(np.unique(markers)) - 2  # -2 para fundo e bordas

    return jsonify({"count": count})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)