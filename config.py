
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATA_FILE = RAW_DATA_DIR / "Online Retail.xlsx"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
RFM_DATA_FILE = PROCESSED_DATA_DIR / "rfm_data.csv"
SEGMENTS_FILE = PROCESSED_DATA_DIR / "customer_segments.csv"

CHURN_MODEL_FILE = MODELS_DIR / "churn_logistic_regression.pkl"
CHURN_FEATURES_FILE = MODELS_DIR / "churn_features.pkl"
