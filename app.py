from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import logging
import json
import mysql.connector
from datetime import datetime

# =============================
# DB CONNECTION
# =============================
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="botaniq"
    )

# =============================
# APP CONFIG
# =============================
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "plant_model.tflite"
LABELS_PATH = "labels.txt"
ADVICE_PATH = "advice.json"
IMG_SIZE = 150
THRESHOLD = 0.60

# =============================
# LOAD LABELS
# =============================
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f if line.strip()]
else:
    LABELS = []
    app.logger.warning("labels.txt not found!")

# =============================
# LOAD ADVICE
# =============================
if os.path.exists(ADVICE_PATH):
    with open(ADVICE_PATH, "r") as f:
        ADVICE = json.load(f)
else:
    ADVICE = {}
    app.logger.warning("advice.json not found!")

# =============================
# LOAD MODEL
# =============================
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Missing TFLite model file.")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =============================
# ROUTES
# =============================
@app.route("/")
def home():
    return jsonify({"message": "Plant Detector API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    try:
        # =============================
        # IMAGE PROCESSING
        # =============================
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(
            input_details[0]['index'],
            img_array.astype(np.float32)
        )
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        # =============================
        # PREDICTION
        # =============================
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        predicted_label = LABELS[class_id]

        if confidence < THRESHOLD:
            return jsonify({
                "class": "unknown",
                "confidence": round(confidence, 4),
                "error": "Gambar tidak dikenali"
            }), 200

        treatment = ADVICE.get(
            predicted_label,
            ["Tidak ada saran perawatan tersedia."]
        )

        # =============================
        # SAVE LOG TO DATABASE ✅
        # =============================
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                """
                INSERT INTO detection_logs (class_name, confidence, created_at)
                VALUES (%s, %s, %s)
                """,
                (predicted_label, confidence, datetime.now())
            )
            db.commit()
            cursor.close()
            db.close()
        except Exception as db_error:
            app.logger.error(f"DB ERROR: {db_error}")

        # =============================
        # RESPONSE
        # =============================
        return jsonify({
            "class": predicted_label,
            "confidence": round(confidence, 4),
            "treatment": treatment
        })

    except Exception as e:
        app.logger.exception("Error in /predict")
        return jsonify({
            "error": "internal error",
            "detail": str(e)
        }), 500

# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
