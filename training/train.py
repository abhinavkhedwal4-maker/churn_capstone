"""Train the churn model and write the artifacts to /model.

Usage (from the repo root):
    python -m training.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, readable_feature

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train")

RANDOM_STATE = 42
MODEL_DIR = Path("model")

COLUMN_RENAMES = {
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
}


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path).rename(columns=COLUMN_RENAMES)

    # TotalCharges is text in the raw file; the 11 blanks are brand-new customers (tenure 0).
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0.0)

    # customerID is dropped (identifier) and gender is deliberately excluded to avoid
    # a model that discriminates on a protected attribute.
    target = (df["Churn"] == "Yes").astype(int)
    return df[ALL_FEATURES], target


def build_pipeline(classifier) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def evaluate(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions)), 4),
        "recall": round(float(recall_score(y_true, predictions)), 4),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y_true, probabilities) -> float:
    """Pick the cut-off that maximises F1. Handles the ~26% churn class imbalance."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


def main(data_path: Path) -> None:
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log.info(
        "Train rows: %d | Test rows: %d | Churn rate: %.1f%%",
        len(X_train),
        len(X_test),
        y.mean() * 100,
    )

    # Baseline first: any real model has to beat this.
    baseline = build_pipeline(LogisticRegression(max_iter=1000))
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(y_test, baseline.predict_proba(X_test)[:, 1], threshold=0.5)
    log.info("Baseline ROC AUC: %.4f", baseline_metrics["roc_auc"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        build_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
        param_distributions={
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [2, 3, 4],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__min_samples_leaf": [1, 5, 20],
        },
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    log.info("Best CV ROC AUC: %.4f | params: %s", search.best_score_, search.best_params_)

    # Tune the threshold on out-of-fold predictions so the test set stays untouched.
    out_of_fold = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, out_of_fold)

    metrics = evaluate(y_test, best.predict_proba(X_test)[:, 1], threshold)
    log.info("Threshold: %.3f | Test metrics: %s", threshold, metrics)

    names = best.named_steps["preprocess"].get_feature_names_out()
    importances = best.named_steps["classifier"].feature_importances_
    top = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": readable_feature(names[i]), "importance": round(float(importances[i]), 4)}
        for i in top
    ]

    trained_at = datetime.now(UTC)
    metadata = {
        "model_version": trained_at.strftime("%Y.%m.%d"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "threshold": round(threshold, 4),
        "cv_roc_auc": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "top_features": top_features,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_DIR / "churn_pipeline.joblib", compress=3)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved model and metadata to %s/", MODEL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    )
    main(parser.parse_args().data)
