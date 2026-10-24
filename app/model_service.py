import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from app.features import ALL_FEATURES, readable_feature
from app.schemas import CustomerFeatures, PredictionResponse, Reason

logger = logging.getLogger(__name__)

TOP_REASONS = 5

RECOMMENDED_ACTIONS = {
    "High": "Prioritise for a retention call and offer a contract upgrade or loyalty discount.",
    "Medium": "Add to a watch-list and send a satisfaction survey or targeted offer.",
    "Low": "No action needed. Keep the customer engaged with regular service.",
}


def recommended_action(risk: str) -> str:
    return RECOMMENDED_ACTIONS.get(
        risk, "No action needed. Keep the customer engaged with regular service."
    )


def risk_level(probability: float, threshold: float) -> str:
    """Bands are relative to the tuned decision threshold, not a hard-coded 0.5."""
    if probability >= threshold:
        return "High"
    if probability >= threshold / 2:
        return "Medium"
    return "Low"


class ChurnPredictor:
    """Loads the trained pipeline once at startup and explains every prediction with SHAP."""

    def __init__(self, pipeline_path: Path, metadata_path: Path) -> None:
        pipeline = joblib.load(pipeline_path)
        self._preprocess = pipeline.named_steps["preprocess"]
        self._classifier = pipeline.named_steps["classifier"]
        self._explainer = shap.TreeExplainer(self._classifier)
        self._feature_names = [
            readable_feature(name) for name in self._preprocess.get_feature_names_out()
        ]

        self.metadata = json.loads(Path(metadata_path).read_text())
        self.threshold = float(self.metadata["threshold"])
        logger.info("Loaded churn model version %s", self.metadata["model_version"])

    def predict(self, customer: CustomerFeatures) -> PredictionResponse:
        frame = pd.DataFrame([customer.model_dump()], columns=ALL_FEATURES)
        features = self._preprocess.transform(frame)

        probability = float(self._classifier.predict_proba(features)[0, 1])
        contributions = self._shap_for_churn(features)

        strongest = np.argsort(np.abs(contributions))[::-1][:TOP_REASONS]
        reasons = [
            Reason(
                feature=self._feature_names[i],
                impact=round(float(contributions[i]), 4),
                direction="increases risk" if contributions[i] > 0 else "decreases risk",
            )
            for i in strongest
        ]

        risk = risk_level(probability, self.threshold)
        return PredictionResponse(
            churn_probability=round(probability, 4),
            will_churn=probability >= self.threshold,
            risk_level=risk,
            threshold=round(self.threshold, 4),
            recommended_action=recommended_action(risk),
            top_reasons=reasons,
            model_version=self.metadata["model_version"],
        )

    def _shap_for_churn(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(self._explainer.shap_values(features))
        if values.ndim == 3:  # some tree models return one slice per class
            values = values[..., 1]
        return values[0]
