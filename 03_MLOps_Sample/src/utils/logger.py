import logging
import os

# Pastikan folder logs ada
os.makedirs("logs", exist_ok=True)

# Konfigurasi logging
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_prediction(input_data: dict, prediction: float):
    logging.info(f"Input={input_data} | Predict={prediction}")