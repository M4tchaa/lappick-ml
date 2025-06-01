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

# --- SISTEM SKOR DENGAN PEMBOBOTAN PER KATEGORI ---
def calculate_score(row, category):
    # Poin dasar untuk setiap komponen
    cpu_points = 0
    gpu_points = 0
    ram_points = 0
    storage_points = 0
    screen_points = 0
    touch_points = 0

    # --- Poin Dasar CPU ---
    cpu_text = str(row.get('CPU', '')).lower()
    if 'core i9' in cpu_text or 'ryzen 9' in cpu_text:
        cpu_points = 100
    elif 'core i7' in cpu_text or 'ryzen 7' in cpu_text or 'intel evo core i7' in cpu_text:
        cpu_points = 70
    elif 'core i5' in cpu_text or 'ryzen 5' in cpu_text or 'intel evo core i5' in cpu_text:
        cpu_points = 45
    elif 'core i3' in cpu_text or 'ryzen 3' in cpu_text:
        cpu_points = 25
    elif 'apple m2' in cpu_text:
        cpu_points = 80
    elif 'apple m1' in cpu_text:
        cpu_points = 60
    elif 'intel celeron' in cpu_text or 'amd athlon' in cpu_text or 'amd 3020e' in cpu_text:
        cpu_points = 5
    elif 'qualcomm snapdragon' in cpu_text:
        cpu_points = 10
    else:
        cpu_points = 2

    # --- Poin Dasar GPU ---
    gpu_text = str(row.get('GPU', '')).lower()
    if any(g in gpu_text for g in ['rtx 4090', 'rtx 4080', 'rx 7900']): gpu_points = 150
    elif any(g in gpu_text for g in ['rtx 4070', 'rtx 3080', 'rx 7800', 'rx 6800']): gpu_points = 120
    elif any(g in gpu_text for g in ['rtx 4060', 'rtx 3070', 'rx 7700', 'rx 6700']): gpu_points = 100
    elif any(g in gpu_text for g in ['rtx 4050', 'rtx 3060', 'rtx 2060', 'rx 7600', 'rx 6600']): gpu_points = 80
    elif any(g in gpu_text for g in ['rtx 3050 ti', 'rtx 3050', 'rtx 2050', 'gtx 1660', 'rx 6500m', 'rx 6500']): gpu_points = 60
    elif any(g in gpu_text for g in ['gtx 1650', 'mx550', 'mx450', 'mx350', 'rx 6400']): gpu_points = 35
    elif 'iris xe' in gpu_text or ('radeon graphics' in gpu_text and not any(x in gpu_text for x in ['rx 6', 'rx 7', '610 m'])):
        gpu_points = 15
    elif '610 m' in gpu_text:
         gpu_points = 12
    elif 'integrated' in gpu_text or not gpu_text:
        gpu_points = 5
    
    # --- Poin Dasar RAM ---
    ram_size_val = 0
    try:
        ram_value = row.get('RAM', '0')
        ram_match = re.search(r'(\d+)', str(ram_value))
        if ram_match:
            ram_size_val = int(ram_match.group(1))
    except ValueError:
        ram_size_val = 0
        
    if ram_size_val >= 32: ram_points = 60
    elif ram_size_val >= 16: ram_points = 40
    elif ram_size_val >= 12: ram_points = 25
    elif ram_size_val >= 8: ram_points = 15
    elif ram_size_val >= 4: ram_points = 5
    
    # --- Poin Dasar Storage ---
    storage_size_gb_val = 0
    try:
        storage_value = row.get('Storage', '0')
        storage_match = re.search(r'(\d+)', str(storage_value))
        if storage_match:
            storage_size_gb_val = int(storage_match.group(1))
            if 'tb' in str(storage_value).lower():
                 storage_size_gb_val *= 1024
    except ValueError:
        storage_size_gb_val = 0

    storage_type_val = str(row.get('Storage type', '')).upper()

    base_storage_size_points = 0
    if storage_size_gb_val >= 2000: base_storage_size_points = 50
    elif storage_size_gb_val >= 1000: base_storage_size_points = 35
    elif storage_size_gb_val >= 512: base_storage_size_points = 20
    elif storage_size_gb_val >= 256: base_storage_size_points = 10
    elif storage_size_gb_val >= 128: base_storage_size_points = 5
    
    storage_type_bonus = 0
    if 'SSD' in storage_type_val:
        storage_type_bonus = 30
    elif 'HDD' in storage_type_val:
        storage_type_bonus = 5
    elif 'EMMC' in storage_type_val:
        storage_type_bonus = 2
        # Penalti eMMC untuk kategori tertentu akan dihandle oleh bobot rendah untuk storage
        # atau bobot negatif jika diperlukan, tapi untuk sekarang bobot rendah sudah cukup.
    storage_points = base_storage_size_points + storage_type_bonus

    # --- Poin Dasar Ukuran Layar ---
    screen_size_val = 0
    try:
        screen_value = row.get('Screen', '0')
        screen_match = re.search(r'(\d+\.?\d*)', str(screen_value))
        if screen_match:
            screen_size_val = float(screen_match.group(1))
    except ValueError:
        screen_size_val = 0

    if screen_size_val >= 17: screen_points = 15
    elif screen_size_val >= 15.6: screen_points = 10
    elif screen_size_val >= 14: screen_points = 7
    elif 13 <= screen_size_val < 14: screen_points = 5 # Portabilitas
    
    # --- Poin Dasar Touchscreen ---
    touch_enabled_val = row.get('Touch', 0)
    if touch_enabled_val == 1 or str(touch_enabled_val).lower() == 'true':
        touch_points = 10

    # --- Bobot Kategori ---
    category_weights = {
        'gaming': {
            'cpu': 1.5,  # Bobot CPU untuk gaming
            'gpu': 2.0,  # Bobot GPU sangat tinggi untuk gaming
            'ram': 1.0,  # Bobot RAM untuk gaming
            'storage': 0.8, # Bobot Storage (kecepatan SSD penting)
            'screen': 0.3, 
            'touch': 0.1   
        },
        'desain': {
            'cpu': 1.5,  # Bobot CPU sangat tinggi untuk desain
            'gpu': 1.2,  # Bobot GPU (akurasi warna, VRAM) penting
            'ram': 1.5,  # Bobot RAM sangat tinggi untuk desain
            'storage': 1.0, 
            'screen': 0.5, # Ukuran dan kualitas layar (meski kualitas tdk ada di data)
            'touch': 0.3   # Bisa berguna untuk desain
        },
        'kantor': {
            'cpu': 1.2,  
            'gpu': 0.2,  # Bobot GPU rendah untuk kantor
            'ram': 1.2,  
            'storage': 1.5, # Kecepatan SSD dan reliabilitas sangat penting
            'screen': 0.4, # Portabilitas dan kenyamanan layar
            'touch': 0.2
        },
        'umum':   { 
            'cpu': 1.0,
            'gpu': 0.5,  
            'ram': 1.0,
            'storage': 1.2, # SSD tetap penting untuk pengalaman umum
            'screen': 0.3,
            'touch': 0.2
        }
    }
    # Gunakan bobot 'umum' sebagai fallback jika kategori tidak ditemukan
    current_weights = category_weights.get(category, category_weights['umum'])

    # Hitung skor akhir dengan pembobotan
    weighted_score = (
        cpu_points * current_weights['cpu'] +
        gpu_points * current_weights['gpu'] +
        ram_points * current_weights['ram'] +
        storage_points * current_weights['storage'] +
        screen_points * current_weights['screen'] +
        touch_points * current_weights['touch']
    )
    
    return weighted_score

# Sistem rekomendasi
def rekomendasi(teks_user, top_n=5):
    if df_laptop.empty:
        return "error_no_data", pd.DataFrame()

    teks_clean = preprocess(teks_user)
    vektor_input = vectorizer.transform([teks_clean]).toarray()
    prediksi_proba = model.predict(vektor_input)[0]
    label_index = np.argmax(prediksi_proba)
    label = label_encoder.inverse_transform([label_index])[0]

    df_filtered = df_laptop.copy()
    
    # Hitung skor untuk setiap laptop di df_filtered
    # calculate_score akan menangani parsing dari kolom asli di 'row'
    df_filtered["skor"] = df_filtered.apply(lambda row: calculate_score(row, label), axis=1)
    
    budget = ekstrak_budget(teks_user)
    if budget:
        if 'Final Price' in df_filtered.columns:
            df_filtered['Final Price'] = pd.to_numeric(df_filtered['Final Price'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['Final Price']) 
            df_filtered = df_filtered[df_filtered['Final Price'] <= budget]
        else:
            print("Peringatan: Kolom 'Final Price' tidak ditemukan untuk filter budget.")

    hasil = df_filtered.sort_values(by=["skor", "Final Price"], ascending=[False, True]).head(top_n)
    
    # Drop kolom 'skor' sebelum mengembalikan hasil
    hasil_cleaned = hasil.drop(columns=['skor'], errors='ignore')

    return label, hasil_cleaned.reset_index(drop=True)

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