"""Data drift detection stubs for production monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between training and serving distributions.

    PSI < 0.1: no drift. 0.1–0.25: moderate. > 0.25: significant drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    def bucket_fractions(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=breakpoints)
        fracs = counts / len(arr)
        return np.clip(fracs, 1e-6, None)

    e_fracs = bucket_fractions(expected)
    a_fracs = bucket_fractions(actual)
    return float(np.sum((a_fracs - e_fracs) * np.log(a_fracs / e_fracs)))


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    psi_threshold: float = 0.1,
) -> dict[str, dict]:
    """Compute PSI for each numeric feature and flag drift.

    Returns:
        Dict mapping feature name to {"psi": float, "drift": bool}.
    """
    report: dict[str, dict] = {}
    for col in numeric_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        psi = compute_psi(reference[col].dropna(), current[col].dropna())
        report[col] = {"psi": round(psi, 4), "drift": psi > psi_threshold}
    return report


def prediction_drift(
    reference_preds: np.ndarray,
    current_preds: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """Compare churn rate between reference and current prediction windows."""
    ref_rate = float(np.mean(reference_preds))
    cur_rate = float(np.mean(current_preds))
    delta = abs(cur_rate - ref_rate)
    return {
        "reference_churn_rate": round(ref_rate, 4),
        "current_churn_rate": round(cur_rate, 4),
        "delta": round(delta, 4),
        "drift": delta > threshold,
    }
