from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
import joblib
import time
import os
import httpx
from datetime import date
from pathlib import Path
import pandas as pd

from src.utils.logger import log_prediction

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = "yudhaislamisulistya"
REPO = "rpl_sc_06_01"
WORKFLOW = "ci-cd-mlops.yaml"

# 🔹 Prometheus imports
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# 1. Load model
model = joblib.load("models/model.pkl")

app = FastAPI(title="API Prediksi Harga Komoditas")

# 🔹 Definisi metrics Prometheus
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)

# 2. Definisikan input & output
class PriceInput(BaseModel):
    price_lag1: float
    price_lag2: float

class PricePrediction(BaseModel):
    predicted_price: float

class ActualPriceInput(BaseModel):
    date: date           # format: YYYY-MM-DD
    price_today: float


DATA_PATH = Path("data/data.csv")

def upsert_actual_price(input_date, price_today):
    # Baca CSV
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Cek apakah tanggal sudah ada
    mask = df["date"] == pd.to_datetime(input_date)

    if mask.any():
        # ✅ Sudah ada → replace price_today saja
        df.loc[mask, "price_today"] = price_today
    else:
        # ✅ Belum ada → append baris baru
        # Cari tanggal sebelumnya (paling dekat di bawahnya)
        prev_df = df[df["date"] < pd.to_datetime(input_date)].sort_values("date")

        if not prev_df.empty:
            last_row = prev_df.iloc[-1]
            price_lag1 = last_row["price_today"]
            price_lag2 = last_row["price_lag1"]
        else:
            # Kalau belum ada data sebelumnya (kasus awal banget)
            price_lag1 = price_today
            price_lag2 = price_today

        new_row = {
            "date": pd.to_datetime(input_date),
            "price_lag1": float(price_lag1),
            "price_lag2": float(price_lag2),
            "price_today": float(price_today),
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Urutkan lagi berdasarkan tanggal
    df = df.sort_values("date")

    # Simpan kembali
    df.to_csv(DATA_PATH, index=False)

    return {
        "date": str(input_date),
        "price_today": float(price_today),
    }

# 🔹 Middleware untuk hitung latency & jumlah request
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    method = request.method
    status_code = response.status_code

    # catat latency
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)

    # catat jumlah request
    REQUEST_COUNT.labels(
        endpoint=endpoint,
        method=method,
        http_status=status_code,
    ).inc()

    return response


# 🔹 Endpoint untuk Prometheus scrape metrics
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# 3. Endpoint uji
@app.get("/")
def root():
    return {"message": "API Prediksi Harga Komoditas aktif"}


# 4. Endpoint prediksi
@app.post("/predict", response_model=PricePrediction)
def predict(data: PriceInput):
    X = [[data.price_lag1, data.price_lag2]]
    y_pred = model.predict(X)[0]

    # Logging ke file
    log_prediction(data.dict(), float(y_pred))

    return PricePrediction(predicted_price=float(y_pred))

@app.post("/set-actual")
def set_actual(data: ActualPriceInput):
    result = upsert_actual_price(
        input_date=data.date,
        price_today=data.price_today,
    )
    return {
        "message": "Actual price saved",
        "data": result,
    }
    
@app.post("/github-auto-train")
async def github_auto_train():
    """
    Endpoint sederhana untuk trigger GitHub Actions workflow.
    Tidak butuh request body.
    """
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN belum diset di environment")

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{WORKFLOW}/dispatches"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
            json={"ref": "main"},
        )

    if resp.status_code >= 300:
        # Kalau ada error dari GitHub, lempar ke client biar kelihatan
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"GitHub API error: {resp.text}",
        )

    return {"status": "ok", "github_status": resp.status_code}