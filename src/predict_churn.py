
import math

import joblib
import pandas as pd

from config import CHURN_MODEL_FILE
from src.logger import get_logger


# Create a logger for this file
logger = get_logger(__name__)

# Features required by our trained model
FEATURES = ["Frequency", "Monetary"]


def predict_customer_churn(customer_data):
    """
    Predict whether a customer belongs to the inactivity-risk class.

    Returns:
        prediction: 0 or 1
        risk_probability: probability of class 1 as a percentage
    """

    logger.info("Customer prediction requested")

    # Step 1: Check that all required features are provided
    for feature in FEATURES:

        if feature not in customer_data:
            logger.warning("A required feature is missing")
            raise ValueError(f"Please provide {feature}.")

        value = customer_data[feature]

        # Step 2: Reject Boolean values such as True and False
        if isinstance(value, bool):
            logger.warning("Invalid input value received")
            raise ValueError(f"{feature} must be a valid number.")

        # Step 3: Convert the value into a number
        try:
            value = float(value)

        except (TypeError, ValueError):
            logger.warning("Input could not be converted to a number")
            raise ValueError(f"{feature} must be a valid number.") from None

        # Step 4: Reject missing, infinite, or negative values
        if not math.isfinite(value) or value < 0:
            logger.warning("Invalid numeric value received")
            raise ValueError(
                f"{feature} must be a finite, non-negative number."
            )

    # Step 5: Check that the trained model exists
    if not CHURN_MODEL_FILE.exists():
        logger.error("The trained churn model was not found")
        raise FileNotFoundError(
            "The trained churn model was not found."
        )

    # Step 6: Load the saved machine learning model
    model = joblib.load(CHURN_MODEL_FILE)

    # Step 7: Prepare customer data in a pandas DataFrame
    customer = pd.DataFrame([
        {
            feature: float(customer_data[feature])
            for feature in FEATURES
        }
    ])

    # Step 8: Predict the customer's class
    prediction = model.predict(customer)[0]

    # Step 9: Calculate the probability of class 1
    probabilities = model.predict_proba(customer)[0]

    # Find the position of class 1 in the model's classes
    class_index = list(model.classes_).index(1)

    # Convert the probability to a percentage
    risk_probability = float(probabilities[class_index] * 100)

    logger.info(
        "Prediction completed: class=%s",
        prediction,
    )

    # Step 10: Return the prediction and probability
    return prediction, risk_probability


# Run this example when the file is executed directly
if __name__ == "__main__":

    # Example customer information
    customer_data = {
        "Frequency": 4,
        "Monetary": 1000,
    }

    # Get the prediction
    prediction, risk_probability = predict_customer_churn(customer_data)

    # Display the result
    if prediction == 1:
        print("Customer is classified as at risk of inactivity.")
    else:
        print("Customer is not classified as at risk of inactivity.")

    print("Prediction:", prediction)
    print(f"Inactivity probability: {risk_probability:.2f}%")
