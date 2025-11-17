from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# CONFIG
MODEL_PATH = "plant_model.tflite"    # pastikan file ini ada di repo Render
LABELS_PATH = "labels.txt"           # file yang kita simpan dari Colab
IMG_SIZE = 150                       # sesuai training (150x150)
# fallback treatment (boleh ganti dengan advice.json jika ada)
TREATMENT = {
    "bacterial_spot": "Gunakan fungisida tembaga dan hindari kelembapan berlebih.",
    "early_blight": "Pangkas daun yang terinfeksi dan semprot fungisida.",
    "late_blight": "Gunakan fungisida berbahan aktif chlorothalonil.",
    "leaf_mold": "Kurangi kelembaban dan tingkatkan ventilasi tanaman.",
    "healthy": "Tanaman sehat! Tetap rawat dengan penyiraman & nutrisi yang baik."
}

# LOAD LABELS (dari file)
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines() if line.strip()]
    app.logger.info(f"Loaded {len(LABELS)} labels from {LABELS_PATH}: {LABELS}")
else:
    # fallback: kosongkan sehingga mismatch akan mudah dilihat
    LABELS = []
    app.logger.warning(f"{LABELS_PATH} not found. Please upload labels.txt generated from Colab.")

# LOAD TFLITE
if not os.path.exists(MODEL_PATH):
    app.logger.error(f"Model file {MODEL_PATH} not found. Put plant_model.tflite in project root.")
    raise SystemExit("Missing TFLite model file.")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
app.logger.info(f"TFLite input: {input_details}, output: {output_details}")

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
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, IMG_SIZE, IMG_SIZE, 3)

        # Pastikan input shape cocok
        # Some TFLite models expect different dtype or shape; do minimal cast:
        input_index = input_details[0]['index']
        # if model expects uint8, convert
        if input_details[0]['dtype'] == np.uint8:
            # scale to 0-255 and cast
            img_array_uint8 = (img_array * 255).astype(np.uint8)
            interpreter.set_tensor(input_index, img_array_uint8)
        else:
            interpreter.set_tensor(input_index, img_array.astype(np.float32))

        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        preds = np.array(preds)  # convert to numpy array

        # Debug logging
        app.logger.info(f"Preds shape: {preds.shape}, preds (first row): {preds[0].tolist() if preds.size>0 else preds}")

        # Validate output dims
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        n_model_classes = preds.shape[1]
        n_labels = len(LABELS)

        if n_labels == 0:
            # no labels file present
            return jsonify({
                "error": "labels.txt missing on server. Upload labels.txt exported from training (Colab).",
                "preds_shape": preds.shape,
                "message": "Generate labels.txt from Colab with: labels = list(train_generator.class_indices.keys()); write to file."
            }), 500

        if n_model_classes != n_labels:
            # MISMATCH: kembalikan info debug agar mudah diperbaiki
            topk = 5
            top_indices = preds[0].argsort()[-topk:][::-1].tolist()
            top_list = [{"index": int(i), "score": float(preds[0][i])} for i in top_indices]
            return jsonify({
                "error": "label/model mismatch",
                "model_num_classes": int(n_model_classes),
                "server_num_labels": int(n_labels),
                "labels_sample": LABELS[:min(10, n_labels)],
                "top_predictions": top_list
            }), 500

        # normal flow: find predicted class
        class_id = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        predicted_label = LABELS[class_id] if class_id < len(LABELS) else None

        treatment = TREATMENT.get(predicted_label, "Tidak ada saran khusus untuk kelas ini.")

        return jsonify({
            "class": predicted_label,
            "confidence": round(confidence, 4),
            "treatment": treatment
        })
    except Exception as e:
        app.logger.exception("Exception during /predict")
        return jsonify({"error": "internal error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
