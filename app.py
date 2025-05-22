from flask import Flask, request, jsonify
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
    if label == 'gaming':
        df = df[(df['RAM'] >= 8) & df['GPU'].str.contains("RTX|GTX|Radeon", na=False, case=False)]
    elif label == 'desain':
        df = df[(df['RAM'] >= 8) & 
                df['Storage'] >= 256 &
                df['CPU'].str.contains("i5|i7|Ryzen 5|Ryzen 7", na=False, case=False) &
                df['GPU'].str.contains("RTX|GTX|Radeon", na=False, case=False)]
    elif label == 'kantor':
        df = df[(df['RAM'] >= 4) & (df['Storage'] >= 256)]
    else:
        df = df[df['RAM'] >= 4]

    budget = ekstrak_budget(teks_user)
    if budget:
        df = df[df['Final Price'] <= budget]

    hasil = df.sort_values(by="Final Price").head(top_n)
    return label, hasil.reset_index(drop=True)

# === Flask App ===
app = Flask(__name__)

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