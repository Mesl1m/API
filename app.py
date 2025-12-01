from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# CONFIG
MODEL_PATH = "plant_model.tflite"    # pastikan file ini ada di repo
LABELS_PATH = "labels.txt"           # pastikan file ini ada
IMG_SIZE = 150                       # sesuai training (150x150)

# --- THRESHOLD UNTUK MENOLAK GAMBAR NON-DAUN ---
THRESHOLD = 0.60   # ubah ke 0.7 / 0.8 jika ingin lebih ketat

# --- OPTIONAL TREATMENT ---
TREATMENT = {
    "bacterial_spot": "Gunakan fungisida tembaga dan hindari kelembapan berlebih.",
    "early_blight": "Pangkas daun yang terinfeksi dan semprot fungisida.",
    "late_blight": "Gunakan fungisida berbahan aktif chlorothalonil.",
    "leaf_mold": "Kurangi kelembaban dan tingkatkan ventilasi tanaman.",
    "healthy": "Tanaman sehat! Tetap rawat dengan penyiraman & nutrisi yang baik."
}

# LOAD LABELS
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines() if line.strip()]
    app.logger.info(f"Loaded labels: {LABELS}")
else:
    LABELS = []
    app.logger.warning("labels.txt not found!")

# LOAD TFLITE MODEL
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Missing TFLite model file.")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
app.logger.info("Model loaded successfully.")

@app.route("/")
def home():
    return jsonify({"message": "Plant Detector API (TFLite) is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    try:
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        input_index = input_details[0]['index']

        # handle uint8 model input
        if input_details[0]['dtype'] == np.uint8:
            img_array_uint8 = (img_array * 255).astype(np.uint8)
            interpreter.set_tensor(input_index, img_array_uint8)
        else:
            interpreter.set_tensor(input_index, img_array.astype(np.float32))

        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        preds = np.array(preds)

        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        # CEK KECOCOKAN LABEL
        if len(LABELS) != preds.shape[1]:
            return jsonify({
                "error": "Label count mismatch with model output!",
                "labels_found": len(LABELS),
                "model_output_classes": preds.shape[1]
            }), 500

        # --- AMBIL PREDIKSI ---
        class_id = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        predicted_label = LABELS[class_id]

        # ================================
        #   THRESHOLD REJECTION SYSTEM
        # ================================
        if confidence < THRESHOLD:
            return jsonify({
                "class": "unknown",
                "confidence": round(confidence, 4),
                "error": "Gambar bukan daun atau tidak bisa dikenali. Mohon upload foto daun yang jelas."
            }), 200

        treatment = TREATMENT.get(predicted_label, "Tidak ada saran khusus.")

        return jsonify({
            "class": predicted_label,
            "confidence": round(confidence, 4),
            "treatment": treatment
        })

    except Exception as e:
        app.logger.exception("Error in /predict")
        return jsonify({"error": "internal error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
