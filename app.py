from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os
from google.cloud import vision
import requests
import json
from groq import Groq

app = Flask(__name__)
CORS(app)

# --- CONFIGURAÇÃO DE CHAVES E AMBIENTE ---

# 1. Configuração da Credencial do Google (Secret File no Render)
# O Render salva arquivos secretos (Secret Files) em /etc/secrets/ automaticamente
google_creds_path = "/etc/secrets/google-vision-key.json"

if os.path.exists(google_creds_path):
    # Se o arquivo existir (ambiente Render), configura a variável de ambiente para ele
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path
else:
    # Se não encontrar (ambiente local), apenas avisa. 
    print(f"Aviso: Arquivo de credenciais não encontrado em {google_creds_path}.")

# 2. Leitura das Chaves de API (Variáveis de Ambiente)
# Estas chaves devem ser configuradas na aba "Environment" do Render
GEMINI_KEY = os.environ.get("GEMINI_KEY")
GROQ_KEY = os.environ.get("GROQ_KEY")

# Verifica se as chaves foram carregadas corretamente
if not GEMINI_KEY:
    print("⚠️ AVISO: GEMINI_KEY não encontrada nas variáveis de ambiente.")
if not GROQ_KEY:
    print("⚠️ AVISO: GROQ_KEY não encontrada nas variáveis de ambiente.")


# --- INICIALIZAÇÃO DOS CLIENTES ---

# O vision_client vai procurar automaticamente a variável GOOGLE_APPLICATION_CREDENTIALS que definimos acima
try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Erro ao iniciar Vision Client (verifique as credenciais): {e}")
    vision_client = None

# Inicializa cliente Groq
# Se a chave não existir, o cliente pode falhar ao ser usado, trataremos isso nas funções
try:
    groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
except Exception as e:
    print(f"Erro ao iniciar Groq Client: {e}")
    groq_client = None

# ==========================================
# FUNÇÃO AUXILIAR: REDIMENSIONAMENTO
# ==========================================
def resize_image(image_bytes, max_dimension=768):
    """
    Redimensiona a imagem mantendo a proporção para economizar tokens.
    """
    try:
        # Converte bytes para imagem OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        
        # Se a imagem já for pequena, não faz nada
        if max(h, w) <= max_dimension:
            return image_bytes

        # Calcula a nova escala mantendo aspect ratio
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensiona
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Codifica de volta para bytes (JPG)
        _, buffer = cv2.imencode('.jpg', resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buffer.tobytes()
        
    except Exception as e:
        print(f"⚠️ Erro ao redimensionar: {e}. Usando original.")
        return image_bytes

# ==========================================
# CAMADA 1: GEMINI 2.5 FLASH
# ==========================================
def call_gemini_primary(image_bytes, item_name):
    if not GEMINI_KEY:
        print("Gemini pulado: chave não configurada.")
        return None

    # Redimensionamos também para o Gemini para ele responder mais rápido
    optimized_image = resize_image(image_bytes, max_dimension=1024)
    
    print(f"1️⃣ Tentando Gemini 2.5 para: {item_name}...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    
    prompt_text = f"""
    Conte a quantidade EXATA de itens que são: "{item_name}".
    Ignore objetos de fundo ou lixo visual.
    Responda APENAS JSON: {{ "count": <int> }}
    """
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(optimized_image).decode()}}
            ]
        }],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            clean = r.json()["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "").strip()
            return int(json.loads(clean).get("count", 0))
        else:
            print(f"Erro na resposta do Gemini: {r.text}")
    except Exception as e:
        print(f"⚠️ Gemini falhou: {e}")
    
    return None

# ==========================================
# CAMADA 2: GROQ / LLAMA 4 VISION
# ==========================================
def call_groq_secondary(image_bytes, item_name):
    if not groq_client:
        print("Groq pulado: cliente ou chave não configurados.")
        return None

    print(f"2️⃣ Tentando Groq (Llama 4 Scout) para: {item_name}...")
    
    # Redimensiona para economizar tokens
    optimized_image = resize_image(image_bytes, max_dimension=768)
    
    base64_image = base64.b64encode(optimized_image).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze the image. Count exactly how many '{item_name}' are present. Output ONLY raw JSON format: {{ \"count\": <number> }}"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            temperature=0.1,
            max_tokens=100, 
            response_format={"type": "json_object"},
        )
        
        result = json.loads(chat_completion.choices[0].message.content)
        return int(result.get("count", 0))
        
    except Exception as e:
        print(f"⚠️ Groq falhou: {e}")
        return None

# ==========================================
# CAMADA 3: VISION API (Fallback)
# ==========================================
def call_vision_fallback(content, user_input):
    if not vision_client:
        print("Google Vision pulado: cliente não inicializado.")
        return None, []

    print("3️⃣ Tentando Google Vision API (Fallback)...")
    # Vision cobra por requisição, não por tamanho
    image = vision.Image(content=content)
    try:
        response = vision_client.object_localization(image=image)
    except Exception as e:
        print(f"Erro na chamada do Vision API: {e}")
        return None, []
    
    detected_objects = []
    
    for obj in response.localized_object_annotations:
        if obj.score < 0.60: continue
        
        # Filtros geométricos básicos
        box = obj.bounding_poly.normalized_vertices
        w = box[2].x - box[0].x
        # Exemplo: Ignora coisas gigantes que ocupam a tela toda (>90%)
        if w > 0.9: continue
        
        detected_objects.append(obj)
        
    return len(detected_objects), detected_objects

# ==========================================
# ROTA PRINCIPAL
# ==========================================
@app.route("/api/count", methods=["POST"])
def count_objects():
    data = request.get_json()
    img_b64 = data.get("image")
    user_input = data.get("item_name", "objeto").strip().lower()
    
    if not img_b64: return jsonify({"error": "no image"}), 400

    try:
        header, encoded = img_b64.split(",", 1)
    except:
        encoded = img_b64
    content = base64.b64decode(encoded)
    
    # --- FLUXO EM CASCATA ---
    
    final_count = None
    provider = "unknown"
    boxes = []

    # 1. Gemini
    final_count = call_gemini_primary(content, user_input)
    if final_count is not None:
        provider = "gemini-2.5-flash"

    # 2. Groq (se Gemini falhou)
    if final_count is None:
        final_count = call_groq_secondary(content, user_input)
        if final_count is not None:
            provider = "groq-llama-3.2"
    
    # 3. Vision API (se ambos falharam)
    if final_count is None:
        final_count, boxes = call_vision_fallback(content, user_input)
        if final_count is not None:
            provider = "google-vision-classic"
        else:
            final_count = 0
            provider = "failed-all"
    
    # --- GERAÇÃO DA RESPOSTA VISUAL ---
    # Usamos a imagem ORIGINAL para desenhar
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    
    if "vision-classic" not in provider and final_count > 0:
        # Desenha número gigante (IAs Generativas)
        text = str(final_count)
        font_scale = 5
        thickness = 12
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        # Sombra preta
        cv2.putText(img, text, (text_x+5, text_y+5), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,0,0), thickness+4)
        # Texto verde
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)
        
    elif "vision-classic" in provider:
        # Desenha caixas (Vision API)
        for obj in boxes:
            box = obj.bounding_poly.normalized_vertices
            x1 = int(box[0].x * img.shape[1])
            y1 = int(box[0].y * img.shape[0])
            x2 = int(box[2].x * img.shape[1])
            y2 = int(box[2].y * img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 8)

    _, buffer = cv2.imencode(".jpg", img)
    img_out = base64.b64encode(buffer).decode()

    print(f"✅ Resultado Final: {final_count} | Provedor: {provider}")
    
    return jsonify({
        "count": final_count if final_count is not None else 0,
        "annotated_image": f"data:image/jpeg;base64,{img_out}",
        "message": f"Contagem: {final_count} (via {provider})",
        "provider": provider
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)