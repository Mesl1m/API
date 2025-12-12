from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import os
import logging
import json
import random

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ======================================================
# CONFIG
# ======================================================
PLANT_MODEL_PATH = "plant_model.tflite"
HUMAN_MODEL_PATH = "bestygkedua.pt"                # YOLO MODEL
LABELS_PATH = "labels.txt"
ADVICE_PLANT_PATH = "advice.json"           # DAUN
ADVICE_HUMAN_PATH = "advicehuman.json"      # MANUSIA
IMG_SIZE = 150

THRESHOLD_PLANT = 0.60
THRESHOLD_HUMAN = 0.50   # YOLO confidence person

# ======================================================
# LOAD YOLO HUMAN DETECTOR
# ======================================================
if not os.path.exists(HUMAN_MODEL_PATH):
    raise SystemExit("Missing YOLO model file best.pt")

human_model = YOLO(HUMAN_MODEL_PATH)
app.logger.info("YOLO model loaded successfully.")

# ======================================================
# LOAD PLANT TFLITE MODEL
# ======================================================
if not os.path.exists(PLANT_MODEL_PATH):
    raise SystemExit("Missing plant TFLite model file.")

interpreter = tf.lite.Interpreter(model_path=PLANT_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ======================================================
# LOAD LABELS
# ======================================================
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines() if line.strip()]
else:
    LABELS = []
    app.logger.warning("labels.txt not found!")

# ======================================================
# LOAD ADVICE FOR PLANT
# ======================================================
if os.path.exists(ADVICE_PLANT_PATH):
    with open(ADVICE_PLANT_PATH, "r") as f:
        ADVICE_PLANT = json.load(f)
else:
    ADVICE_PLANT = {}
    app.logger.warning("advice.json (plant) not found!")

# ======================================================
# LOAD ADVICE FOR HUMAN
# ======================================================
if os.path.exists(ADVICE_HUMAN_PATH):
    with open(ADVICE_HUMAN_PATH, "r") as f:
        ADVICE_HUMAN = json.load(f)
else:
    ADVICE_HUMAN = {}
    app.logger.warning("advicehuman.json not found!")

# ======================================================
# RANDOM HUMAN ADVICE
# ======================================================
def get_random_human_advice():
    if not ADVICE_HUMAN:
        return {"type": "unknown", "messages": ["No human advice available."]}

    category = random.choice(list(ADVICE_HUMAN.keys()))
    messages = ADVICE_HUMAN[category]

    return {
        "type": category,
        "messages": messages
    }

# ======================================================
# FUNGSI CEK MANUSIA (YOLO)
# ======================================================
def predict_human(image_path):
    result = human_model(image_path)[0]
    best_conf = 0

    for box in result.boxes:
        cls = int(box.cls[0])     # class id
        conf = float(box.conf[0]) # confidence

        if cls == 0 and conf > best_conf:  # class 0 = person
            best_conf = conf

    return best_conf

# ======================================================
# FUNGSI CEK DAUN (TFLITE)
# ======================================================
def predict_plant(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if input_details[0]['dtype'] == np.uint8:
        img_array = (img_array * 255).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    class_id = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    label = LABELS[class_id] if class_id < len(LABELS) else "unknown"

    return label, confidence

# ======================================================
# ROUTE HOME
# ======================================================
@app.route("/")
def home():
    return jsonify({"message": "Human & Plant Detector API is running."})

# ======================================================
# ROUTE PREDICT
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    try:
        file = request.files["file"]
        img_path = "/tmp/uploaded.jpg"
        file.save(img_path)

        # 1. CHECK HUMAN FIRST
        human_score = predict_human(img_path)

        if human_score >= THRESHOLD_HUMAN:
            return jsonify({
                "result": "human",
                "confidence": round(human_score, 4),
                "advice": get_random_human_advice()
            })

        # 2. CHECK PLANT
        plant_label, plant_conf = predict_plant(img_path)

        if plant_conf >= THRESHOLD_PLANT:
            return jsonify({
                "result": "plant",
                "class": plant_label,
                "confidence": round(plant_conf, 4),
                "treatment": ADVICE_PLANT.get(plant_label, ["Tidak ada saran perawatan."])
            })

        # 3. UNKNOWN
        return jsonify({
            "result": "unknown",
            "human_confidence": round(human_score, 4),
            "plant_confidence": round(plant_conf, 4)
        })

    except Exception as e:
        app.logger.exception("ERROR in /predict")
        return jsonify({"error": str(e)}), 500

# ======================================================
# RUN SERVER — FIX FOR RENDER
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
