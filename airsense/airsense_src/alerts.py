"""Alert helpers for severe AQI hours."""
from __future__ import annotations

import pandas as pd

from airsense_src.health_advisory import aqi_to_category

SEVERE_CATEGORIES = {"Poor", "Very Poor", "Severe"}


def flag_severe_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where AQI is Poor or worse, tagged with a severity flag."""
    if df.empty:
        result = df.copy()
        result["is_severe_hour"] = pd.Series(dtype=bool)
        result["aqi_category"] = pd.Series(dtype=object)
        return result

    result = df.copy()
    if "aqi_category" not in result.columns:
        result["aqi_category"] = result["aqi"].apply(aqi_to_category)
    result["is_severe_hour"] = result["aqi_category"].isin(SEVERE_CATEGORIES)
    severe = result[result["is_severe_hour"]].copy()
    severe["is_severe_hour"] = True
    return severe


def summarize_alerts(df: pd.DataFrame) -> dict:
    """Summarize the number and share of severe AQI hours."""
    if df.empty:
        return {
            "flagged_hours": 0,
            "total_hours": 0,
            "flagged_share_pct": 0.0,
            "peak_flagged_hours": 0,
            "peak_flagged_share_pct": 0.0,
        }

    working = df.copy()
    if "aqi_category" not in working.columns:
        working["aqi_category"] = working["aqi"].apply(aqi_to_category)
    working["is_severe_hour"] = working["aqi_category"].isin(SEVERE_CATEGORIES)

    flagged = int(working["is_severe_hour"].sum())
    total = int(len(working))
    peak_flagged = int((working["is_severe_hour"] & working.get("is_peak_hour", False)).sum())
    return {
        "flagged_hours": flagged,
        "total_hours": total,
        "flagged_share_pct": round((flagged / total * 100.0) if total else 0.0, 2),
        "peak_flagged_hours": peak_flagged,
        "peak_flagged_share_pct": round((peak_flagged / flagged * 100.0) if flagged else 0.0, 2),
    }
