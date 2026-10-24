import pytest

from app.config import Settings, get_settings
from app.main import app

HIGH_RISK = {
    "tenure": 1,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "monthly_charges": 95.0,
    "total_charges": 95.0,
}
LOW_RISK = {
    "tenure": 60,
    "contract": "Two year",
    "internet_service": "DSL",
    "payment_method": "Credit card (automatic)",
    "monthly_charges": 55.0,
    "total_charges": 3300.0,
}


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_a_complete_prediction(client, customer_payload):
    response = client.post("/predict", json=customer_payload)
    assert response.status_code == 200

    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert len(body["top_reasons"]) == 5


def test_risky_profile_scores_higher_than_loyal_profile(client, customer_payload):
    high = client.post("/predict", json={**customer_payload, **HIGH_RISK}).json()
    low = client.post("/predict", json={**customer_payload, **LOW_RISK}).json()
    assert high["churn_probability"] > low["churn_probability"]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("tenure", "twelve"),
        ("tenure", "12"),
        ("tenure", -3),
        ("tenure", 12.5),
        ("contract", "Weekly"),
        ("monthly_charges", -1),
        ("senior_citizen", 2),
    ],
)
def test_invalid_values_are_rejected(client, customer_payload, field, bad_value):
    customer_payload[field] = bad_value
    assert client.post("/predict", json=customer_payload).status_code == 422


def test_missing_field_is_rejected(client, customer_payload):
    del customer_payload["tenure"]
    assert client.post("/predict", json=customer_payload).status_code == 422


def test_unknown_field_is_rejected(client, customer_payload):
    customer_payload["favourite_colour"] = "blue"
    assert client.post("/predict", json=customer_payload).status_code == 422


def test_impossible_service_combination_is_rejected(client, customer_payload):
    customer_payload["internet_service"] = "No"  # add-ons still say Yes/No
    assert client.post("/predict", json=customer_payload).status_code == 422


def test_api_key_is_enforced_when_configured(client, customer_payload):
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret")

    assert client.post("/predict", json=customer_payload).status_code == 401
    wrong = client.post("/predict", json=customer_payload, headers={"X-API-Key": "wrong"})
    assert wrong.status_code == 401
    right = client.post("/predict", json=customer_payload, headers={"X-API-Key": "secret"})
    assert right.status_code == 200


def test_model_info_exposes_metrics(client):
    body = client.get("/model/info").json()
    assert body["metrics"]["roc_auc"] > 0.7
    assert 0 < body["threshold"] < 1


def test_security_headers_are_set(client):
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
