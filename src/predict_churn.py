
import joblib
import pandas as pd

model = joblib.load("models/churn_adaboost.pkl")
features = joblib.load("models/churn_features.pkl")

customers = {
    "Recency": 120,
    "Frequency": 4,
    "Monetary": 1000,
    "Total_Quantity": 23,
    "Unique_Products": 5,
    "Purchase_Months": 6,
    "Customer_Lifetime": 120,
    "Avg_Order_Value": 14.0,
    "Average_Days_Between_Orders": 40.0
}


def predict_customer_churn(customer_data):
    customer_df = pd.DataFrame([customer_data])
    customer_df = customer_df[features]

    prediction = model.predict(customer_df)
    prediction_prob = model.predict_proba(customer_df)

    class_1_index = list(model.classes_).index(1)
    churn_probability = prediction_prob[0][class_1_index] * 100

    return prediction, churn_probability


result, churn_probability = predict_customer_churn(customers)

if result[0] == 1:
    print("The Customer is at risk of churn")
else:
    print("The Customer is not at risk of churn")

print("Prediction:", result)
print(f"Churn Probability: {churn_probability:.2f}%")
