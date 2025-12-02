import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    print("🔄 Memulai proses training model...")

    # 1. Load Data
    try:
        df = pd.read_csv('Walmart.csv')
    except FileNotFoundError:
        print("❌ Error: File 'Walmart.csv' tidak ditemukan.")
        return

    # 2. Feature Selection
    features = [
        'inventory_level',      # Stok saat ini
        'reorder_point',        # Batas minimal pesan ulang
        'supplier_lead_time',   # Lama tunggu supplier
        'forecasted_demand'     # Prediksi permintaan sistem lama
    ]
    target = 'stockout_indicator'

    # 3. Preprocessing
    # Pastikan target adalah integer (0/1)
    df[target] = df[target].astype(int)

    X = df[features]
    y = df[target]

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Scaling (Penting untuk Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train Model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 7. Evaluasi Singkat
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model berhasil dilatih. Akurasi: {acc*100:.2f}%")

    # 8. Simpan Model dan Scaler
    output_file = 'walmart_stockout_model.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f"💾 Model dan Scaler disimpan ke '{output_file}'")

if __name__ == "__main__":
    train_and_save_model()
