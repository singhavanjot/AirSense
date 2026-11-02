"""Cleaning and alignment utilities for AirSense."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PEAK_HOUR_WINDOWS, PROCESSED_DATA_DIR


def _ensure_timestamp_series(series: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamp


def _remove_aqi_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "aqi" not in frame.columns:
        return frame

    working = frame.copy()
    valid = working["aqi"].dropna()
    if valid.empty:
        return working
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = min(500.0, q3 + 1.5 * iqr)
    return working[(working["aqi"].isna()) | ((working["aqi"] >= lower) & (working["aqi"] <= upper))].copy()


def clean_aqi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean AQI data by removing invalid values and interpolating gaps."""
    if df.empty:
        return df.copy()

    working = df.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = _ensure_timestamp_series(working["timestamp"])
    if "city" not in working.columns:
        working["city"] = "Unknown"

    if "aqi" in working.columns:
        working["aqi"] = pd.to_numeric(working["aqi"], errors="coerce")
        working.loc[(working["aqi"] < 0) | (working["aqi"] == 999), "aqi"] = np.nan
        working = _remove_aqi_outliers(working)

    numeric_candidates = [
        column
        for column in ["aqi", "pm25_ugm3", "temperature_c", "humidity_pct", "wind_speed_mps", "precipitation_mm"]
        if column in working.columns
    ]
    for column in working.columns:
        if pd.api.types.is_numeric_dtype(working[column]) and column not in numeric_candidates:
            numeric_candidates.append(column)

    cleaned_groups: list[pd.DataFrame] = []
    for city, group in working.groupby("city", dropna=False):
        group = group.sort_values("timestamp").drop_duplicates(subset=["timestamp", "city"])
        if "timestamp" in group.columns:
            group = group.set_index("timestamp")
        for column in numeric_candidates:
            if column in group.columns:
                group[column] = pd.to_numeric(group[column], errors="coerce")
                group[column] = group[column].interpolate(method="time", limit_direction="both")
                group[column] = group[column].ffill().bfill()
        if "aqi" in group.columns:
            group = group[(group["aqi"].isna()) | ((group["aqi"] >= 0) & (group["aqi"] <= 500))]
        group = group.reset_index()
        cleaned_groups.append(group)

    cleaned = pd.concat(cleaned_groups, ignore_index=True) if cleaned_groups else pd.DataFrame(columns=working.columns)
    cleaned = cleaned.dropna(subset=["timestamp", "city", "aqi"]) if "aqi" in cleaned.columns else cleaned
    if "aqi" in cleaned.columns:
        cleaned["aqi"] = cleaned["aqi"].round(0)
    return cleaned.sort_values(["city", "timestamp"]).reset_index(drop=True)


def merge_aqi_weather(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge AQI and weather data on a consistent hourly UTC timestamp."""
    if aqi_df.empty:
        return aqi_df.copy()

    aqi = aqi_df.copy()
    weather = weather_df.copy()
    aqi["timestamp"] = _ensure_timestamp_series(aqi["timestamp"])
    if not weather.empty:
        weather["timestamp"] = _ensure_timestamp_series(weather["timestamp"])
    else:
        weather = pd.DataFrame(columns=["timestamp", "city", "temperature_c", "humidity_pct", "wind_speed_mps", "precipitation_mm", "source"])

    aqi["timestamp"] = aqi["timestamp"].dt.floor("h")
    weather["timestamp"] = weather["timestamp"].dt.floor("h")

    if weather.empty:
        merged = aqi.copy()
        for column in ["temperature_c", "humidity_pct", "wind_speed_mps", "precipitation_mm", "source_weather"]:
            if column not in merged.columns:
                merged[column] = np.nan
        return merged.sort_values(["city", "timestamp"]).reset_index(drop=True)

    merge_keys = ["city", "timestamp"] if "city" in aqi.columns and "city" in weather.columns else ["timestamp"]
    merged = pd.merge_asof(
        aqi.sort_values("timestamp"),
        weather.sort_values("timestamp"),
        on="timestamp",
        by="city" if "city" in merge_keys else None,
        direction="nearest",
        tolerance=pd.Timedelta("1H"),
        suffixes=("_aqi", "_weather"),
    )
    return merged.sort_values(["city", "timestamp"]).reset_index(drop=True)


def tag_peak_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Tag rows as peak or off-peak using the configured traffic windows."""
    if df.empty:
        result = df.copy()
        result["is_peak_hour"] = pd.Series(dtype=bool)
        result["peak_period"] = pd.Series(dtype=object)
        return result

    working = df.copy()
    working["timestamp"] = _ensure_timestamp_series(working["timestamp"])
    working["hour"] = working["timestamp"].dt.hour
    working["day_of_week"] = working["timestamp"].dt.dayofweek
    working["is_weekend"] = working["day_of_week"] >= 5

    morning_start, morning_end = PEAK_HOUR_WINDOWS["morning"]
    evening_start, evening_end = PEAK_HOUR_WINDOWS["evening"]
    weekday_mask = working["day_of_week"].isin(PEAK_HOUR_WINDOWS["weekdays"])
    morning_mask = working["hour"].between(morning_start, morning_end, inclusive="both")
    evening_mask = working["hour"].between(evening_start, evening_end, inclusive="both")
    weekend_start, weekend_end = PEAK_HOUR_WINDOWS["weekend"]
    weekend_mask = working["is_weekend"] & working["hour"].between(weekend_start, weekend_end, inclusive="both")

    working["is_peak_hour"] = (weekday_mask & (morning_mask | evening_mask)) | weekend_mask
    working["peak_period"] = np.where(working["is_peak_hour"], "peak", "off_peak")
    return working


def save_cleaned_output(df: pd.DataFrame, city: str) -> Path:
    """Persist a cleaned city dataset to the processed data directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_city = "".join(char.lower() if char.isalnum() else "_" for char in city).strip("_")
    path = PROCESSED_DATA_DIR / f"{safe_city}_cleaned.csv"
    df.to_csv(path, index=False)
    return path
