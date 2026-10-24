import pytest

from app.config import get_settings
from app.model_service import ChurnPredictor, risk_level
from app.schemas import CustomerFeatures


@pytest.fixture(scope="module")
def predictor() -> ChurnPredictor:
    settings = get_settings()
    return ChurnPredictor(settings.pipeline_path, settings.metadata_path)


def test_reasons_are_sorted_by_absolute_impact(predictor, customer_payload):
    result = predictor.predict(CustomerFeatures(**customer_payload))
    impacts = [abs(reason.impact) for reason in result.top_reasons]
    assert impacts == sorted(impacts, reverse=True)


def test_prediction_is_deterministic(predictor, customer_payload):
    customer = CustomerFeatures(**customer_payload)
    first = predictor.predict(customer)
    second = predictor.predict(customer)
    assert first.churn_probability == second.churn_probability


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.9, "High"), (0.45, "High"), (0.3, "Medium"), (0.1, "Low")],
)
def test_risk_level_bands(probability, expected):
    assert risk_level(probability, threshold=0.4) == expected
