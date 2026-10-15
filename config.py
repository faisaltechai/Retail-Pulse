
import os
from pathlib import Path

from dotenv import load_dotenv


# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the project's .env file
load_dotenv(BASE_DIR / ".env")


# Application settings
APP_NAME = os.getenv("APP_NAME", "RetailPulse")
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model and log directories
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"


# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "Online Retail.xlsx"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
RFM_DATA_FILE = PROCESSED_DATA_DIR / "rfm_data.csv"
SEGMENTS_FILE = PROCESSED_DATA_DIR / "customer_segments.csv"


# Trained model files
CHURN_MODEL_FILE = MODELS_DIR / "churn_logistic_regression.pkl"
CHURN_FEATURES_FILE = MODELS_DIR / "churn_features.pkl"


# Create required directories if they don't exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
