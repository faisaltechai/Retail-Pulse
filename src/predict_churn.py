import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/churn_logistic_regression.pkl")

# Features used by the model
features = ["Frequency", "Monetary"]


def predict_customer_churn(customer_data):
    customer_df = pd.DataFrame([customer_data])

    # Select the required features
    customer_df = customer_df[features]

    # Make prediction
    prediction = model.predict(customer_df)[0]
    probabilities = model.predict_proba(customer_df)[0]

    # Get churn probability for class 1
    class_1_index = list(model.classes_).index(1)
    churn_probability = probabilities[class_1_index] * 100

    return prediction, churn_probability


if __name__ == "__main__":
    # Sample customer data
    customer = {
        "Frequency": 4,
        "Monetary": 1000
    }

    # Predict customer churn
    result, churn_probability = predict_customer_churn(customer)

    if result == 1:
        print("The customer is at risk of churn.")
    else:
        print("The customer is not predicted to be at risk of churn.")

    print("Prediction:", result)
    print(f"Churn Probability: {churn_probability:.2f}%")