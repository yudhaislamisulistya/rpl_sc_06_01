from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schema import (
    PredictRequest, PredictResponse
)
from app.model import ModelService, MODEL_NAME

app = FastAPI(
    title="Melbourne House Price Prediction API",
    description="An API to predict house prices in Melbourne.",
    version=MODEL_NAME,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_NAME}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        prediction = model_service.predict_one(req.model_dump())
        return PredictResponse(prediction=prediction, model_version=MODEL_NAME)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    