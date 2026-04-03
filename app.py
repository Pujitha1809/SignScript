"""
Sign Language to Text Translator — Flask Backend
"""
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import json
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── Config ───────────────────────────────────────────────────────────────────
# POINTING BACK TO THE ORIGINAL MODEL
MODEL_PATH  = r"model\sign_model_savedmodel.keras"
LABELS_PATH = os.environ.get("LABELS_PATH", r"model\class_labels.json")
IMG_SIZE    = 64
CONF_THRESH = 0.60

# ─── Load Model ───────────────────────────────────────────────────────────────
print("USING MODEL:", MODEL_PATH)
logger.info("Loading model from %s …", MODEL_PATH)

import keras

# Keras 3 wrapper to intercept the legacy quantization key
class SafeDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

# Load the model with our SafeDense wrapper
model = keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'Dense': SafeDense}, 
    compile=False
)

logger.info("Model loaded successfully.")

# Load class labels
try:
    with open(LABELS_PATH) as f:
        class_indices = json.load(f)
        idx_to_label  = {int(v): k for k, v in class_indices.items()}
    logger.info("Labels loaded. Classes: %s", list(idx_to_label.values()))
except Exception as e:
    logger.error("Could not load class labels! Error: %s", e)
    idx_to_label = {}

# ─── MediaPipe Hands ──────────────────────────────────────────────────────────
from mediapipe.python.solutions import hands as mp_hands

hands_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def decode_frame(b64_data: str) -> np.ndarray:
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    arr       = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def extract_hand_roi(frame_bgr: np.ndarray):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results   = hands_model.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    h, w = frame_bgr.shape[:2]
    lm   = results.multi_hand_landmarks[0]
    xs   = [p.x for p in lm.landmark]
    ys   = [p.y for p in lm.landmark]

    pad = 0.15
    x1  = max(0, int((min(xs) - pad) * w))
    y1  = max(0, int((min(ys) - pad) * h))
    x2  = min(w, int((max(xs) + pad) * w))
    y2  = min(h, int((max(ys) + pad) * h))

    roi       = frame_bgr[y1:y2, x1:x2]
    landmarks = {"bbox": [x1, y1, x2, y2]}
    return roi, landmarks

def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = rgb.astype(np.float32) / 255.0
    return np.expand_dims(norm, axis=0)

def predict(roi_bgr: np.ndarray) -> dict:
    inp    = preprocess_roi(roi_bgr)
    probs  = model.predict(inp, verbose=0)[0]
    top3_i = probs.argsort()[-3:][::-1]

    top3 = [
        {"label": idx_to_label.get(i, f"Unknown-{i}"), "confidence": float(probs[i])}
        for i in top3_i
    ]
    best = top3[0]
    return {
        "letter":     best["label"] if best["confidence"] >= CONF_THRESH else None,
        "confidence": best["confidence"],
        "accepted":   best["confidence"] >= CONF_THRESH,
        "top3":       top3,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' field"}), 400

    try:
        frame = decode_frame(data["frame"])
    except Exception as e:
        return jsonify({"error": f"Could not decode frame: {e}"}), 400

    roi, landmarks = extract_hand_roi(frame)

    if roi is None or roi.size == 0:
        return jsonify({
            "hand_detected": False,
            "letter":        None,
            "confidence":    0.0,
            "accepted":      False,
            "top3":          [],
            "bbox":          None,
        })

    result = predict(roi)
    result["hand_detected"] = True
    result["bbox"]          = landmarks["bbox"] if landmarks else None
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": len(idx_to_label), "model": MODEL_PATH})

# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)