"""Pandera schema tests for the Telco churn dataset.

These tests validate that data loaded by DataLoader conforms to expected
types, value ranges, and business constraints — before any preprocessing.
"""

import pytest
from pandera.pandas import Column, DataFrameSchema, Check

from src.data import DataLoader
from src.config import get_config


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

RAW_SCHEMA = DataFrameSchema(
    columns={
        # Numeric features
        "Tenure Months": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(72),
            ],
            nullable=False,
        ),
        "Monthly Charges": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "Total Charges": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        # Target
        "Churn Value": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
    },
    strict=False,  # allow extra columns not listed here
    coerce=True,
)

BINARY_COLS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
]

MULTICLASS_COLS = {
    "Multiple Lines": {"No", "Yes", "No phone service"},
    "Internet Service": {"DSL", "Fiber optic", "No"},
    "Online Security": {"No", "Yes", "No internet service"},
    "Online Backup": {"No", "Yes", "No internet service"},
    "Device Protection": {"No", "Yes", "No internet service"},
    "Tech Support": {"No", "Yes", "No internet service"},
    "Streaming TV": {"No", "Yes", "No internet service"},
    "Streaming Movies": {"No", "Yes", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "Payment Method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_df():
    config = get_config("development")
    loader = DataLoader(config)
    df = loader.load_dataset()
    loader.identify_target()
    return df


# ---------------------------------------------------------------------------
# Schema tests (marked with @pytest.mark.schema)
# ---------------------------------------------------------------------------

@pytest.mark.schema
def test_pandera_numeric_and_target(loaded_df):
    """Numeric columns and target conform to schema constraints."""
    RAW_SCHEMA.validate(loaded_df)


@pytest.mark.schema
def test_tenure_range(loaded_df):
    """Tenure Months must be between 0 and 72."""
    col = loaded_df["Tenure Months"].astype(float)
    assert col.min() >= 0, f"Tenure min={col.min()} < 0"
    assert col.max() <= 72, f"Tenure max={col.max()} > 72"


@pytest.mark.schema
def test_monthly_charges_positive(loaded_df):
    """Monthly Charges must be non-negative."""
    col = loaded_df["Monthly Charges"].astype(float)
    assert col.min() >= 0, f"Monthly Charges has negative values: min={col.min()}"


@pytest.mark.schema
def test_total_charges_non_negative(loaded_df):
    """Total Charges must be non-negative (after coercion)."""
    col = loaded_df["Total Charges"].astype(float)
    assert (col >= 0).all(), "Total Charges has negative values"


@pytest.mark.schema
def test_target_binary(loaded_df):
    """Churn Value must be binary (0 or 1)."""
    target = loaded_df["Churn Value"].astype(int)
    assert set(target.unique()).issubset({0, 1}), \
        f"Target contains unexpected values: {target.unique()}"


@pytest.mark.schema
def test_churn_rate_plausible(loaded_df):
    """Churn rate should be between 10% and 50% (Telco benchmark range)."""
    rate = loaded_df["Churn Value"].astype(int).mean()
    assert 0.10 <= rate <= 0.50, f"Churn rate {rate:.2%} out of expected range"


@pytest.mark.schema
def test_no_missing_values(loaded_df):
    """Dataset must have no null values after loading."""
    nulls = loaded_df.isnull().sum()
    assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"


@pytest.mark.schema
def test_no_duplicates(loaded_df):
    """Dataset must have no duplicate rows after loading."""
    dups = loaded_df.duplicated().sum()
    assert dups == 0, f"Found {dups} duplicate rows"


@pytest.mark.schema
def test_leakage_columns_absent(loaded_df):
    """Data leakage columns must be absent from the loaded dataset."""
    leak_cols = [
        "Churn Score", "Churn Reason", "CLTV",
        "CustomerID", "Zip Code", "Latitude", "Longitude", "City",
        "Lat Long", "Count", "Churn Label",
    ]
    present = [c for c in leak_cols if c in loaded_df.columns]
    assert not present, f"Leakage columns still present: {present}"


@pytest.mark.schema
def test_multiclass_categorical_values(loaded_df):
    """Categorical columns must only contain known category values."""
    for col, allowed in MULTICLASS_COLS.items():
        if col not in loaded_df.columns:
            continue
        actual = set(loaded_df[col].astype(str).unique())
        unexpected = actual - allowed
        assert not unexpected, \
            f"Column '{col}' has unexpected values: {unexpected}"


@pytest.mark.schema
def test_min_row_count(loaded_df):
    """Dataset must have at least 5,000 rows (Telco dataset has 7,043)."""
    assert len(loaded_df) >= 5000, f"Dataset too small: {len(loaded_df)} rows"


@pytest.mark.schema
def test_feature_count(loaded_df):
    """Dataset must have at least 15 feature columns after leakage removal."""
    feature_cols = [c for c in loaded_df.columns if c != "Churn Value"]
    assert len(feature_cols) >= 15, \
        f"Too few features: {len(feature_cols)} (expected >= 15)"
