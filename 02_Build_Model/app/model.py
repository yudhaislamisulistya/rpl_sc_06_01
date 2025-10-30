import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_melb.pkl"
MODEL_NAME = "melb-v1"
MELB_FEATURES = ["Rooms", "Bathrooms", "LandSize", "BuildingArea", "YearBuilt"]

class ModelService:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)
    
    def _row_from_dict(self, d: dict):
        return [d[f] for f in MELB_FEATURES]
        # [{Rooms: 2, Bathrooms: 1, LandSize: 500, BuildingArea: 150, YearBuilt: 1990}]

    def predict_one(self, payload: dict) -> float:
        X = np.array([self._row_from_dict(payload)], dtype=float)
        # [[2, 1, 500, 150, 1990]]
        yhat = self.model.predict(X)
        # [750000.0] -> 750000.0
        return float(yhat[0]) # 750000.0