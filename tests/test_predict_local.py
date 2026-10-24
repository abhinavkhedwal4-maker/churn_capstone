import json

from app.model_service import recommended_action, risk_level
from scripts.predict_local import SAMPLE_CUSTOMER, main


def test_predict_endpoint_returns_recommended_action(client, customer_payload):
    response = client.post("/predict", json=customer_payload)
    assert response.status_code == 200
    body = response.json()
    assert "recommended_action" in body
    expected_action = recommended_action(body["risk_level"])
    assert body["recommended_action"] == expected_action


def test_medium_risk_band_and_action():
    threshold = 0.4
    # Medium risk: threshold / 2 <= probability < threshold
    prob = 0.25
    level = risk_level(prob, threshold)
    assert level == "Medium"
    action = recommended_action(level)
    assert "watch-list" in action.lower()


def test_predict_local_smoke_default(capsys):
    main([])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["risk_level"] in {"Low", "Medium", "High"}
    assert "recommended_action" in payload
    assert len(payload["top_reasons"]) == 5


def test_predict_local_file_argument(tmp_path, capsys):
    sample_file = tmp_path / "customer.json"
    sample_file.write_text(json.dumps(SAMPLE_CUSTOMER), encoding="utf-8")
    main(["--file", str(sample_file)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "churn_probability" in payload
    assert payload["recommended_action"] == recommended_action(payload["risk_level"])
