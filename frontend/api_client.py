"""Thin HTTP client for the churn API, used by the Streamlit dashboard."""

from __future__ import annotations

import requests


class ApiError(Exception):
    """Raised when an API call fails. The message is safe to show to end users."""


class ChurnApiClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    def health(self) -> bool:
        try:
            return self._request("GET", "/health", timeout=3.0).get("status") == "ok"
        except ApiError:
            return False

    def predict(self, payload: dict) -> dict:
        return self._request("POST", "/predict", json=payload)

    def model_info(self) -> dict:
        return self._request("GET", "/model/info")

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self._base_url}{path}"
        kwargs.setdefault("timeout", self._timeout)
        try:
            response = self._session.request(method, url, **kwargs)
        except requests.Timeout as exc:
            raise ApiError(
                "The API took too long to respond. It may be waking up - try again shortly."
            ) from exc
        except requests.RequestException as exc:
            raise ApiError("Could not reach the API.") from exc

        if response.status_code == 401:
            raise ApiError("The API rejected the API key.")
        if response.status_code == 422:
            raise ApiError(f"Invalid input: {_first_validation_message(response)}")
        if response.status_code == 429:
            raise ApiError("Too many requests - please wait a moment.")
        if not response.ok:
            raise ApiError(f"The API returned an error ({response.status_code}).")

        try:
            return response.json()
        except Exception as exc:
            raise ApiError("The API returned an invalid response.") from exc


def _first_validation_message(response) -> str:
    try:
        detail = response.json()["detail"][0]
        location = ".".join(str(part) for part in detail["loc"] if part != "body")
        return f"{location}: {detail['msg']}"
    except (ValueError, KeyError, IndexError, TypeError):
        return "the API could not validate the request"
