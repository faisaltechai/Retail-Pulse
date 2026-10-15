
import logging

from config import (BASE_DIR,RAW_DATA_FILE,CLEANED_DATA_FILE,RFM_DATA_FILE,SEGMENTS_FILE,CHURN_MODEL_FILE,)
from src.predict_churn import predict_customer_churn


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def check_files():
    required_files = [RAW_DATA_FILE,CLEANED_DATA_FILE,RFM_DATA_FILE,SEGMENTS_FILE,CHURN_MODEL_FILE,]

    for file in required_files:
        if not file.exists():
            logging.error("Required file not found: %s", file)
            return False

    logging.info("All required files are available.")
    return True


def main():
    setup_logging()

    
    logging.info("Project folder: %s", BASE_DIR)

    if not check_files():
        logging.error("Please check the missing files before continuing.")
        return

    customer = {
        "Frequency": 4,
        "Monetary": 1000,
    }

    prediction, probability = predict_customer_churn(customer)

    if prediction == 1:
        logging.info("Customer is classified as at risk of inactivity.")
    else:
        logging.info("Customer is not classified as at risk of inactivity.")

    logging.info("Prediction: %s", prediction)
    logging.info("Inactivity probability: %.2f%%", probability)
    logging.info("RetailPulse finished successfully.")


if __name__ == "__main__":
    main()
