from unittest.mock import MagicMock, patch

import pytest
import requests

from api_client import ApiError, ChurnApiClient


def fake_response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.ok = status < 400
    response.json.return_value = body or {}
    return response


@pytest.fixture
def api() -> ChurnApiClient:
    return ChurnApiClient("http://api.test", api_key="k")


def test_predict_returns_json_body(api):
    reply = fake_response(200, {"risk_level": "Low"})
    with patch.object(api._session, "request", return_value=reply):
        assert api.predict({"tenure": 1}) == {"risk_level": "Low"}


def test_validation_error_names_the_field(api):
    body = {"detail": [{"loc": ["body", "tenure"], "msg": "Input should be a valid integer"}]}
    with patch.object(api._session, "request", return_value=fake_response(422, body)):
        with pytest.raises(ApiError, match="tenure"):
            api.predict({})


@pytest.mark.parametrize(("status", "text"), [(401, "API key"), (429, "Too many"), (500, "500")])
def test_error_statuses_become_friendly_messages(api, status, text):
    with patch.object(api._session, "request", return_value=fake_response(status)):
        with pytest.raises(ApiError, match=text):
            api.model_info()


def test_connection_failure_becomes_api_error(api):
    with patch.object(api._session, "request", side_effect=requests.ConnectionError):
        with pytest.raises(ApiError, match="Could not reach"):
            api.predict({})


def test_health_is_false_when_api_is_down(api):
    with patch.object(api._session, "request", side_effect=requests.ConnectionError):
        assert api.health() is False
