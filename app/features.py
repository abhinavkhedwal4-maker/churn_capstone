"""Feature definitions shared by training and the API so the two never drift apart."""

NUMERIC_FEATURES = ["senior_citizen", "tenure", "monthly_charges", "total_charges"]

CATEGORICAL_FEATURES = [
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _label(column: str) -> str:
    return column.replace("_", " ").capitalize()


def readable_feature(raw_name: str) -> str:
    """Turn a transformer output name into something a business user can read.

    'cat__internet_service_Fiber optic' -> 'Internet service = Fiber optic'
    'num__monthly_charges'              -> 'Monthly charges'
    """
    kind, separator, name = raw_name.partition("__")
    if not separator:
        return _label(raw_name)

    if kind == "cat":
        for column in CATEGORICAL_FEATURES:
            prefix = f"{column}_"
            if name.startswith(prefix):
                return f"{_label(column)} = {name[len(prefix):]}"
    return _label(name)
