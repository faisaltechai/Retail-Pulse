
import logging

from config import (
    BASE_DIR,
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    RFM_DATA_FILE,
    SEGMENTS_FILE,
    CHURN_MODEL_FILE,
)
from src.predict_churn import predict_customer_churn


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def check_files():
    # Check that all required project files are available
    required_files = [
        RAW_DATA_FILE,
        CLEANED_DATA_FILE,
        RFM_DATA_FILE,
        SEGMENTS_FILE,
        CHURN_MODEL_FILE,
    ]

    missing_files = [
        file for file in required_files if not file.exists()
    ]

    if missing_files:
        for file in missing_files:
            logging.error("File not found: %s", file)

        return False

    logging.info("All required files are available.")
    return True


def main():
    setup_logging()

    logging.info("Starting RetailPulse...")
    logging.info("Project folder: %s", BASE_DIR)

    if not check_files():
        logging.error("Please fix the missing files and try again.")
        return

    customer = {
        "Frequency": 4,
        "Monetary": 1000,
    }

    try:
        prediction, probability = predict_customer_churn(customer)

        if prediction == 1:
            logging.info("Customer is at risk of inactivity.")
        else:
            logging.info("Customer is not classified as at risk of inactivity.")

        logging.info("Prediction: %s", prediction)
        logging.info("Inactivity probability: %.2f%%", probability)

    except (ValueError, FileNotFoundError) as error:
        logging.error("Could not make prediction: %s", error)

    logging.info("RetailPulse finished.")


if __name__ == "__main__":
    main()
