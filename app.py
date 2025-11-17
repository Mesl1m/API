from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load model AI
MODEL_PATH = "plant_model.h5"  # ganti sesuai nama file model-mu
model = tf.keras.models.load_model(MODEL_PATH)

# Daftar label sesuai training
labels = ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "healthy"]

# Saran perawatan
disease_advice = {
    "bacterial_spot": "Potong bagian daun yang sakit dan semprot fungisida.",
    "early_blight": "Jaga kelembaban tanah, semprot fungisida preventif.",
    "late_blight": "Buang daun yang terinfeksi dan gunakan fungisida.",
    "leaf_mold": "Tingkatkan sirkulasi udara, semprot larutan fungisida.",
    "healthy": "Tanaman sehat. Lanjutkan perawatan rutin."
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file di request"}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # Preprocessing gambar agar sesuai input model
    img = image.load_img(filepath, target_size=(224, 224))  # sesuaikan ukuran modelmu
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalisasi

    # Prediksi
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds[0])
    prediction = labels[predicted_index]
    advice = disease_advice.get(prediction, "Tidak ada saran perawatan.")
    
    return jsonify({
        "prediction": prediction,
        "advice": advice
    })

if __name__ == '__main__':
    app.run(debug=True)
