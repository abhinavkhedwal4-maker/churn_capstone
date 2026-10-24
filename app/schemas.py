from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

ADDON_FIELDS = (
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
)


class CustomerFeatures(BaseModel):
    """A single customer profile, validated before it ever reaches the model."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    senior_citizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(strict=True, ge=0, le=120, description="Months as a customer")
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: InternetAddon
    online_backup: InternetAddon
    device_protection: InternetAddon
    tech_support: InternetAddon
    streaming_tv: InternetAddon
    streaming_movies: InternetAddon
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0, le=500)
    total_charges: float = Field(ge=0, le=100_000)

    @model_validator(mode="after")
    def check_service_consistency(self) -> Self:
        """Reject combinations that cannot exist in real billing data."""
        has_phone = self.phone_service == "Yes"
        if has_phone == (self.multiple_lines == "No phone service"):
            raise ValueError("multiple_lines is inconsistent with phone_service")

        has_internet = self.internet_service != "No"
        for name in ADDON_FIELDS:
            if has_internet == (getattr(self, name) == "No internet service"):
                raise ValueError(f"{name} is inconsistent with internet_service")
        return self


class Reason(BaseModel):
    feature: str
    impact: float = Field(description="SHAP value in log-odds; positive pushes towards churn")
    direction: Literal["increases risk", "decreases risk"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(ge=0, le=1)
    will_churn: bool
    risk_level: Literal["Low", "Medium", "High"]
    threshold: float
    recommended_action: str
    top_reasons: list[Reason]
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
