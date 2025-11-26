import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

DATA_PATH = "data/data.csv"
MODEL_PATH = "models/model.pkl"

# alamat Pushgateway (bisa diatur via env var)
PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "localhost:9091")

import pandas as pd
from datetime import timedelta


def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model belum ada! Latih dulu dengan train.py")
        return

    df = pd.read_csv(DATA_PATH)
    X = df[["price_lag1", "price_lag2"]]
    y = df["price_today"]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    print(f"📌 Current MAE (full dataset): {mae:.2f}")

    # Simpan juga ke file log (opsional, tetap dipertahankan)
    os.makedirs("logs", exist_ok=True)
    with open("logs/metrics.log", "a") as f:
        f.write(f"MAE={mae:.2f}\n")

    # 🔹 Kirim metric ke Prometheus Pushgateway
    try:
        registry = CollectorRegistry()
        mae_gauge = Gauge(
            "model_mae_current",
            "Current MAE of komoditas model on full dataset",
            registry=registry,
        )
        mae_gauge.set(mae)

        push_to_gateway(
            PUSHGATEWAY_ADDR,
            job="komoditas-model-eval",
            registry=registry,
        )

        print(f"✅ MAE pushed to Pushgateway at {PUSHGATEWAY_ADDR}")
    except Exception as e:
        print(f"⚠️ Gagal push ke Pushgateway: {e}")

if __name__ == "__main__":
    evaluate_model()