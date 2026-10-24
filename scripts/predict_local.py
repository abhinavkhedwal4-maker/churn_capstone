"""Run offline customer churn inference using the serialized pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.model_service import ChurnPredictor
from app.schemas import CustomerFeatures

SAMPLE_CUSTOMER = {
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn from local artifacts.")
    parser.add_argument("--file", type=Path, default=None, help="Path to customer JSON payload.")
    args = parser.parse_args(argv)

    if args.file:
        payload = json.loads(args.file.read_text(encoding="utf-8"))
    else:
        payload = SAMPLE_CUSTOMER

    customer = CustomerFeatures(**payload)
    settings = get_settings()
    predictor = ChurnPredictor(settings.pipeline_path, settings.metadata_path)
    result = predictor.predict(customer)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
