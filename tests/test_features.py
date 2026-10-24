import pytest

from app.features import readable_feature


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("num__tenure", "Tenure"),
        ("num__monthly_charges", "Monthly charges"),
        ("cat__contract_Two year", "Contract = Two year"),
        ("cat__internet_service_Fiber optic", "Internet service = Fiber optic"),
        ("cat__online_security_No internet service", "Online security = No internet service"),
    ],
)
def test_readable_feature(raw, expected):
    assert readable_feature(raw) == expected
