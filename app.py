from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model saat server start
model_path = os.path.join("model", "plant_model.h5")  # <-- sesuaikan nama model
model = load_model(model_path)

# Label sesuai model
labels = ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "healthy"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join("model", file.filename)
    file.save(file_path)

    try:
        img = image.load_img(file_path, target_size=(224, 224))  # sesuaikan input model
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_label = labels[np.argmax(prediction)]

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)  # hapus file sementara

    return jsonify({"prediction": predicted_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
