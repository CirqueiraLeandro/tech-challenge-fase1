"""Sklearn pipeline for Telco churn feature preprocessing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


def build_feature_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build a sklearn Pipeline for numeric scaling + categorical encoding.

    Args:
        numeric_cols: Column names to scale with StandardScaler.
        categorical_cols: Column names to encode with OrdinalEncoder.

    Returns:
        Fitted-ready sklearn Pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode binary target column to 0/1 integers."""
    le = LabelEncoder()
    return le.fit_transform(y).astype(int), le
