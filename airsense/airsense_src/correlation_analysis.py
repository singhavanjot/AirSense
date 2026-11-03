"""Correlation analysis for peak traffic windows and AQI."""
from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from airsense_src.data_cleaning import tag_peak_hours


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "is_peak_hour" not in working.columns and "timestamp" in working.columns:
        working = tag_peak_hours(working)
    if "aqi" in working.columns:
        working["aqi"] = pd.to_numeric(working["aqi"], errors="coerce")
    return working.dropna(subset=["aqi", "is_peak_hour"]) if "aqi" in working.columns else working


def _interpret_strength(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "inconclusive"
    magnitude = abs(value)
    if magnitude < 0.1:
        return "very weak"
    if magnitude < 0.3:
        return "weak"
    if magnitude < 0.5:
        return "moderate"
    return "strong"


def compute_peak_offpeak_stats(df: pd.DataFrame) -> dict:
    """Compute summary stats and a significance test for peak vs. off-peak AQI."""
    working = _prepare_frame(df)
    if working.empty or working["is_peak_hour"].nunique() < 2:
        return {
            "peak": {},
            "off_peak": {},
            "test_used": "insufficient_data",
            "statistic": None,
            "p_value": None,
            "interpretation": "Not enough variation to compare peak and off-peak hours.",
        }

    peak = working[working["is_peak_hour"]]
    off_peak = working[~working["is_peak_hour"]]
    peak_values = peak["aqi"].dropna()
    off_values = off_peak["aqi"].dropna()
    if len(peak_values) < 3 or len(off_values) < 3:
        return {
            "peak": {
                "count": int(len(peak_values)),
                "mean": float(peak_values.mean()) if len(peak_values) else None,
                "median": float(peak_values.median()) if len(peak_values) else None,
            },
            "off_peak": {
                "count": int(len(off_values)),
                "mean": float(off_values.mean()) if len(off_values) else None,
                "median": float(off_values.median()) if len(off_values) else None,
            },
            "test_used": "insufficient_data",
            "statistic": None,
            "p_value": None,
            "interpretation": "Not enough observations in one or both groups.",
        }

    peak_normal = len(peak_values) >= 3 and len(peak_values) <= 5000 and stats.shapiro(peak_values[:5000]).pvalue > 0.05
    off_normal = len(off_values) >= 3 and len(off_values) <= 5000 and stats.shapiro(off_values[:5000]).pvalue > 0.05

    if peak_normal and off_normal:
        test = stats.ttest_ind(peak_values, off_values, equal_var=False, nan_policy="omit")
        test_used = "welch_ttest"
    else:
        test = stats.mannwhitneyu(peak_values, off_values, alternative="two-sided")
        test_used = "mannwhitney_u"

    mean_delta = float(peak_values.mean() - off_values.mean())
    interpretation = "Peak-hour AQI is not meaningfully different from off-peak AQI."
    if test.pvalue < 0.05:
        direction = "higher" if mean_delta > 0 else "lower"
        interpretation = f"Peak-hour AQI is statistically significant and {direction} than off-peak hours."

    return {
        "peak": {
            "count": int(len(peak_values)),
            "mean": float(peak_values.mean()),
            "median": float(peak_values.median()),
        },
        "off_peak": {
            "count": int(len(off_values)),
            "mean": float(off_values.mean()),
            "median": float(off_values.median()),
        },
        "mean_delta": mean_delta,
        "test_used": test_used,
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "interpretation": interpretation,
    }


def partial_correlation_controlling_weather(df: pd.DataFrame) -> dict:
    """Compute the partial correlation between peak hours and AQI controlling for weather."""
    working = _prepare_frame(df)
    covariates = [
        column
        for column in ["temperature_c", "humidity_pct", "wind_speed_mps", "precipitation_mm"]
        if column in working.columns
    ]
    if working.empty or working["is_peak_hour"].nunique() < 2 or not covariates:
        return {
            "correlation": None,
            "p_value": None,
            "n": int(len(working)),
            "covariates": covariates,
            "method": "residual_regression",
            "interpretation": "Inconclusive because the data do not vary enough or weather covariates are missing.",
        }

    analysis = working[["aqi", "is_peak_hour", *covariates]].dropna().copy()
    if len(analysis) < 5 or analysis["is_peak_hour"].nunique() < 2:
        return {
            "correlation": None,
            "p_value": None,
            "n": int(len(analysis)),
            "covariates": covariates,
            "method": "residual_regression",
            "interpretation": "Inconclusive because the cleaned sample is too small or lacks variation.",
        }

    x = analysis["is_peak_hour"].astype(float).to_numpy().reshape(-1, 1)
    y = analysis["aqi"].astype(float).to_numpy()
    z = analysis[covariates].astype(float).to_numpy()

    model_y = LinearRegression().fit(z, y)
    model_x = LinearRegression().fit(z, x.ravel())
    resid_y = y - model_y.predict(z)
    resid_x = x.ravel() - model_x.predict(z)

    if np.allclose(resid_x.std(ddof=1), 0) or np.allclose(resid_y.std(ddof=1), 0):
        return {
            "correlation": None,
            "p_value": None,
            "n": int(len(analysis)),
            "covariates": covariates,
            "method": "residual_regression",
            "interpretation": "Inconclusive because the residual variance collapsed after controlling for weather.",
        }

    correlation, p_value = stats.pearsonr(resid_x, resid_y)
    interpretation = "The adjusted peak-hour relationship is weak or inconclusive."
    if abs(correlation) >= 0.3 and p_value < 0.05:
        direction = "higher" if correlation > 0 else "lower"
        interpretation = f"After controlling for weather, peak hours are associated with {direction} AQI with a meaningful adjusted correlation."
    elif p_value < 0.05:
        direction = "higher" if correlation > 0 else "lower"
        interpretation = f"After controlling for weather, peak hours are associated with {direction} AQI, but the effect is modest."

    return {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "n": int(len(analysis)),
        "covariates": covariates,
        "method": "residual_regression",
        "interpretation": interpretation,
    }
