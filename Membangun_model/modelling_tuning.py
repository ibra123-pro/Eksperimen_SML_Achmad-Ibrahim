import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_model_tuning():
    # Menentukan lokasi tracking lokal MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("UMKM_Pondok_Gede_Modelling")
    
    print("=== Memulai Proses Hyperparameter Tuning ===")
    
    # 1. Load Dataset (Gunakan jalur aman yang sesuai dengan folder VS Code kamu)
    data_path = "data_penjualan_siap_ml.csv" 
    
    try:
        df = pd.read_csv(data_path)
        print(f"[INFO] Dataset berhasil dimuat. Ukuran: {df.shape}")
    except FileNotFoundError:
        # Cadangan jika file .csv sudah kamu copy langsung ke dalam folder Membangun_model
        data_path = "data_penjualan_siap_ml.csv"
        df = pd.read_csv(data_path)
        print(f"[INFO] Dataset berhasil dimuat dari folder lokal. Ukuran: {df.shape}")

    # 2. Memisahkan Fitur (X) dan Target (y)
    # Mencari kolom target penjualan secara otomatis
    target_column = 'Total_Penjualan'
    if target_column not in df.columns:
        target_column = [col for col in df.columns if 'Total' in col][0]
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data menjadi Train & Test (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Definisikan Ruang Parameter (Hyperparameter Space) untuk Tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    print("[INFO] Menjalankan GridSearchCV...")
    base_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Ambil model terbaik hasil tuning
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"[INFO] Parameter Terbaik Ditemukan: {best_params}")
    
    # Evaluasi Model Terbaik pada Data Test
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # 4. MANUAL LOGGING KE MLFLOW (Kriteria Wajib: Menggantikan Autolog)
    with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):
        
        print("[MLFLOW] Melakukan manual logging untuk parameter terbaik...")
        # Mencatat parameter secara manual satu per satu
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        # Mencatat info tambahan seperti nama estimator
        mlflow.log_param("estimator_name", "RandomForestRegressor")
        
        print("[MLFLOW] Melakukan manual logging untuk metrik evaluasi...")
        # Mencatat metrik hasil evaluasi yang setara dengan autolog
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        
        print("[MLFLOW] Menyimpan model artifact secara manual...")
        # Mencatat dan menyimpan file model .pkl secara manual ke dalam MLflow Artifacts
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")
        
        print("\n=== [SUKSES] Tuning selesai dan seluruh log berhasil disimpan secara manual! ===")
        print(f"Hasil Akhir - R2 Score: {r2:.4f} | MSE: {mse:.4f}")

if __name__ == "__main__":
    train_model_tuning()