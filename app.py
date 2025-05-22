from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load model dan komponen
model = tf.keras.models.load_model("lappick_model.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
df_laptop = pd.read_csv("cleaned_dataset.csv")

# Preprocessing
stemmer = StemmerFactory().create_stemmer()
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return stemmer.stem(text)

def ekstrak_budget(teks):
    match = re.search(r'(\d+)\s*(juta|jt)', teks.lower())
    if match:
        return int(match.group(1)) * 1_000_000
    return None

# Sistem rekomendasi
def rekomendasi(teks_user, top_n=5):
    teks_clean = preprocess(teks_user)
    vektor_input = vectorizer.transform([teks_clean]).toarray()
    prediksi = model.predict(vektor_input)[0]
    label_index = np.argmax(prediksi)
    label = label_encoder.inverse_transform([label_index])[0]

    df = df_laptop.copy()

    # GPU keywords untuk deteksi gaming/desain
    gpu_keywords = r"RTX|GTX|Radeon|GeForce"

    # 🎮 Gaming
    if label == 'gaming':
        df = df[
            (df['RAM'] >= 8) &
            (df['Storage'] >= 512) &
            (df['GPU'].str.contains(gpu_keywords, na=False, case=False)) &
            (df['CPU'].str.contains("i5|i7|i9|Ryzen 5|Ryzen 7|Ryzen 9", na=False, case=False))
        ]

        # Tambahkan skor (semakin tinggi semakin bagus)
        def skor(row):
            base = 0
            if "RTX 40" in str(row["GPU"]): base += 5
            elif "RTX 30" in str(row["GPU"]): base += 4
            elif "GTX" in str(row["GPU"]): base += 3
            elif "Radeon" in str(row["GPU"]): base += 2
            base += row["RAM"] / 8
            base += row["Storage"] / 512
            return base

        df["skor"] = df.apply(skor, axis=1)

    elif label == 'desain':
        df = df[
            (df['RAM'] >= 16) &
            (df['Storage'] >= 512) &
            (df['GPU'].str.contains(gpu_keywords, na=False, case=False)) &
            (df['CPU'].str.contains("i7|i9|Ryzen 7|Ryzen 9", na=False, case=False))
        ]
        df["skor"] = df["RAM"] / 8 + df["Storage"] / 512

    elif label == 'kantor':
        df = df[
            (df['RAM'] >= 8) &
            (df['Storage'] >= 256)
        ]
        df["skor"] = df["RAM"] / 4 + df["Storage"] / 256

    else:  # umum
        df = df[
            (df['RAM'] >= 4) &
            (df['Storage'] >= 128)
        ]
        df["skor"] = df["RAM"] / 4 + df["Storage"] / 128

    # 💰 Budget
    budget = ekstrak_budget(teks_user)
    if budget:
        df = df[df['Final Price'] <= budget]

    # 🔼 Urutkan berdasarkan skor (bukan harga)
    hasil = df.sort_values(by="skor", ascending=False).head(top_n)
    return label, hasil.drop(columns="skor").reset_index(drop=True)

# === Flask App ===
app = Flask(__name__)
CORS(app)
@app.route("/rekomendasi", methods=["GET"])
def api_rekomendasi():
    teks = request.args.get("teks")
    if not teks:
        return jsonify({"error": "Parameter 'teks' wajib diisi"}), 400

    label, hasil = rekomendasi(teks)
    return jsonify({
        "input_user": teks,
        "label_prediksi": label,
        "rekomendasi": hasil.to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)

# Sample run
# /rekomendasi?teks=saya ingin laptop untuk main Valorant dengan budget 10 juta