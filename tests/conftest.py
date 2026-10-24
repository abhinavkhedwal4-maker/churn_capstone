import copy

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

_CUSTOMER = {
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 80.5,
    "total_charges": 966.0,
}


@pytest.fixture
def customer_payload() -> dict:
    return copy.deepcopy(_CUSTOMER)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _default_settings():
    """Run every test with auth disabled, whatever is in the developer's .env file."""
    app.dependency_overrides[get_settings] = lambda: Settings(api_key=None)
    yield
    app.dependency_overrides.clear()
