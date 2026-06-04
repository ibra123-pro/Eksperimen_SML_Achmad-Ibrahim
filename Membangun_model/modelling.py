import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    # 1. Mengaktifkan MLflow Autolog Khusus Scikit-Learn (Kriteria Wajib)
    mlflow.sklearn.autolog()
    
    # Menentukan lokasi tracking lokal (akan membuat folder mlruns otomatis jika belum ada)
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("UMKM_Pondok_Gede_Modelling")
    
    print("=== Memulai Proses Training Model ===")
    
    # 2. Load Dataset Hasil Preprocessing Sebelumnya
    # Menyesuaikan dengan letak file data kamu (bisa selevel atau di subfolder)
    data_path = "data_penjualan_siap_ml.csv" 
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset berhasil dimuat. Ukuran: {df.shape}")
    except FileNotFoundError:
        # Fallback jika posisinya ditaruh satu folder yang sama sesuai image_4d020b.png
        print("[ERROR] File tidak ditemukan di folder preprocessing!")
        print("Pastikan di folder '../preprocessing/' sudah ada file 'data_penjualan_siap_ml.csv'")
        return

    # 3. Memisahkan Fitur (X) dan Target (y)
    # Contoh: Mencari tahu pola Total_Penjualan (Sesuaikan target kolom dengan eksperimenmu)
    target_column = 'Total_Penjualan' 
    
    if target_column not in df.columns:
        # Fallback jika kolom target ter-encode atau berbeda nama
        target_column = [col for col in df.columns if 'Total' in col or 'Kategori' in col][0]
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data menjadi Train & Test (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Melatih Model dalam Mlflow Context
    with mlflow.start_run(run_name="RandomForest_Baseline_Local"):
        print("Melatih model RandomForestRegressor tanpa hyperparameter tuning...")
        
        # Inisialisasi model default (baseline)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi & Evaluasi Tambahan
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        print("=== Seluruh parameter, metrik, dan artifak model dicatat otomatis oleh Autolog! ===")

if __name__ == "__main__":
    train_model()