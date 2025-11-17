from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# ===========================
# LOAD TFLITE MODEL
# ===========================
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 150  # sesuaikan dengan training
LABELS = [
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "healthy"
]

@app.route("/")
def home():
    return jsonify({"message": "Plant Disease API is running!"})

# ===========================
# PREDICT ENDPOINT
# ===========================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]

    # Convert file → image
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0

    # reshape menjadi input TFLite
    img_array = np.expand_dims(img_array, axis=0)

    # Masukkan input
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Prediksi
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Rekomendasi perawatan
    treatment = {
        "bacterial_spot": "Gunakan fungisida tembaga dan hindari kelembapan berlebih.",
        "early_blight": "Pangkas daun yang terinfeksi dan semprot fungisida.",
        "late_blight": "Gunakan fungisida berbahan aktif chlorothalonil.",
        "leaf_mold": "Kurangi kelembaban dan tingkatkan ventilasi tanaman.",
        "healthy": "Tanaman sehat! Tetap rawat dengan penyiraman & nutrisi yang baik."
    }

    result = {
        "class": LABELS[class_id],
        "confidence": round(confidence, 4),
        "treatment": treatment[LABELS[class_id]]
    }

    return jsonify(result)

# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
