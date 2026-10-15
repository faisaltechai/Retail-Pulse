
import pytest

from src.predict_churn import predict_customer_churn


def test_prediction_with_valid_data():
    customer = {
        "Frequency": 4,
        "Monetary": 1000
    }

    prediction, probability = predict_customer_churn(customer)

    assert prediction in [0, 1]
    assert 0 <= probability <= 100


def test_missing_customer_data():
    customer = {
        "Frequency": 4
    }

    with pytest.raises(ValueError):
        predict_customer_churn(customer)


def test_negative_customer_data():
    customer = {
        "Frequency": -1,
        "Monetary": 1000
    }

    with pytest.raises(ValueError):
        predict_customer_churn(customer)


def test_invalid_customer_data():
    customer = {
        "Frequency": "hello",
        "Monetary": 1000
    }

    with pytest.raises(ValueError):
        predict_customer_churn(customer)
