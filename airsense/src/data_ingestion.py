"""Data ingestion for AirSense: AQI, weather, and synthetic fallbacks."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from config import (
    CPCB_API_KEY_ENV,
    CPCB_API_URL_ENV,
    DEFAULT_CPCB_API_URL,
    DEFAULT_OPENAQ_API_URL,
    DEFAULT_OPENWEATHER_API_URL,
    OPENAQ_API_KEY_ENV,
    OPENAQ_API_URL_ENV,
    OPENWEATHER_API_KEY_ENV,
    OPENWEATHER_API_URL_ENV,
    RAW_DATA_DIR,
    RANDOM_SEED,
    SOURCE_NAME_CPCB,
    SOURCE_NAME_OPENAQ,
    SOURCE_NAME_OPENWEATHER,
    SOURCE_NAME_SYNTHETIC,
    get_city_config,
)

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _coerce_datetime(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _date_range_label(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    return f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"


def _raw_output_path(city: str, source: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Path:
    return RAW_DATA_DIR / f"{_slugify(city)}_{source}_{_date_range_label(start_date, end_date)}.csv"


def _save_raw_pull(df: pd.DataFrame, city: str, source: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Path:
    path = _raw_output_path(city, source, start_date, end_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _retry_call(factory: Callable[[], pd.DataFrame], attempts: int = 3, base_delay: float = 0.5) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = factory()
            if isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame(result)
        except Exception as exc:  # pragma: no cover - defensive logging
            last_error = exc
            wait_time = base_delay * (2**attempt)
            logger.warning("Attempt %s failed: %s", attempt + 1, exc)
            if attempt < attempts - 1:
                time.sleep(wait_time)
    if last_error is not None:
        logger.error("All retries failed: %s", last_error)
    return pd.DataFrame()


def _seed_for(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    payload = f"{city}|{start_date.isoformat()}|{end_date.isoformat()}|{RANDOM_SEED}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def _pm25_to_aqi(pm25: float) -> float:
    return float(np.clip(pm25 * 4.0 + 10.0, 0.0, 500.0))


def generate_synthetic_data(city: str, start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
    """Generate realistic synthetic AQI and weather time series for a city."""
    start_ts = _coerce_datetime(start_date).floor("h")
    end_ts = _coerce_datetime(end_date).ceil("h")
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    hours = pd.date_range(start_ts, end_ts, freq="h", tz="UTC")
    if hours.empty:
        hours = pd.date_range(start_ts, periods=1, freq="h", tz="UTC")

    city_offset = {
        "Ludhiana": 182.0,
        "Chandigarh": 118.0,
        "Amritsar": 156.0,
    }.get(city, 140.0)
    city_scale = {
        "Ludhiana": 1.15,
        "Chandigarh": 0.95,
        "Amritsar": 1.05,
    }.get(city, 1.0)

    rng = np.random.default_rng(_seed_for(city, start_ts, end_ts))
    hour = hours.hour.to_numpy()
    day_of_year = hours.dayofyear.to_numpy()
    weekday = hours.dayofweek.to_numpy()

    daily_cycle = 18 * np.sin((hour - 7) / 24 * 2 * np.pi) + 12 * np.sin((hour - 18) / 24 * 2 * np.pi)
    seasonal_cycle = 10 * np.sin((day_of_year - 80) / 365 * 2 * np.pi)
    weekend_adjustment = np.where(weekday >= 5, -12.0, 0.0)
    commute_peak = np.where(((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20)), 24.0, 0.0)
    humidity = np.clip(58 + 18 * np.sin((day_of_year + hour) / 29.0) + rng.normal(0, 10, size=len(hours)), 20, 100)
    temperature = np.clip(24 + 11 * np.sin((day_of_year - 170) / 365 * 2 * np.pi) + 6 * np.sin((hour - 14) / 24 * 2 * np.pi) + rng.normal(0, 2.5, len(hours)), -2, 48)
    wind_speed = np.clip(2.5 + 1.7 * np.cos((day_of_year + hour) / 18.0) + rng.normal(0, 0.8, len(hours)), 0.2, 12.0)
    precipitation = np.where(rng.random(len(hours)) < 0.08, np.clip(rng.gamma(shape=1.5, scale=2.0, size=len(hours)), 0, 16), 0.0)

    aqi = (
        city_offset
        + city_scale * (daily_cycle + seasonal_cycle + commute_peak)
        + weekend_adjustment
        + 0.45 * (humidity - 55)
        - 6.5 * wind_speed
        - 4.0 * precipitation
        + rng.normal(0, 12, len(hours))
    )
    aqi = np.clip(aqi, 15, 500)
    pm25 = np.clip((aqi - 10.0) / 4.0 + rng.normal(0, 5, len(hours)), 3, 250)

    station = get_city_config(city)
    return pd.DataFrame(
        {
            "timestamp": hours,
            "city": city,
            "station_id": station["cpcb_station_id"],
            "aqi": np.round(aqi, 0),
            "pm25_ugm3": np.round(pm25, 1),
            "temperature_c": np.round(temperature, 1),
            "humidity_pct": np.round(humidity, 1),
            "wind_speed_mps": np.round(wind_speed, 1),
            "precipitation_mm": np.round(precipitation, 1),
            "source": SOURCE_NAME_SYNTHETIC,
        }
    )


def _fetch_cpcb_aqi(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    api_url = os.getenv(CPCB_API_URL_ENV, DEFAULT_CPCB_API_URL)
    api_key = os.getenv(CPCB_API_KEY_ENV, "").strip()
    if not api_key or "placeholder" in api_url:
        return pd.DataFrame()

    params = {
        "api-key": api_key,
        "format": "json",
        "city": city,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }

    def _request() -> pd.DataFrame:
        response = requests.get(api_url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("records") or payload.get("data") or payload.get("results") or []
        frame = pd.DataFrame(records)
        if frame.empty:
            return frame
        timestamp_column = next((col for col in ["timestamp", "date", "datetime", "recorded_at"] if col in frame.columns), None)
        aqi_column = next((col for col in ["aqi", "value", "aqi_value", "pollution_index"] if col in frame.columns), None)
        if timestamp_column is None or aqi_column is None:
            return pd.DataFrame()
        output = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce"),
                "city": city,
                "aqi": pd.to_numeric(frame[aqi_column], errors="coerce"),
                "station_id": frame["station_id"] if "station_id" in frame.columns else [get_city_config(city)["cpcb_station_id"]] * len(frame),
                "source": SOURCE_NAME_CPCB,
            }
        )
        return output.dropna(subset=["timestamp", "aqi"])

    return _retry_call(_request)


def _fetch_openaq_aqi(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    api_url = os.getenv(OPENAQ_API_URL_ENV, DEFAULT_OPENAQ_API_URL)
    api_key = os.getenv(OPENAQ_API_KEY_ENV, "").strip()
    headers = {"accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    is_v3 = "/v3/" in api_url

    def _request() -> pd.DataFrame:
        params = {
            "date_from": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 1000,
            "sort": "desc",
        }
        if is_v3:
            locations_url = api_url.replace("/measurements", "/locations", 1)
            locations_response = requests.get(
                locations_url,
                params={"city": city, "limit": 100},
                headers=headers,
                timeout=20,
            )
            locations_response.raise_for_status()
            locations = locations_response.json().get("results") or []
            location_ids = [str(item["id"]) for item in locations if item.get("id") is not None]
            if not location_ids:
                return pd.DataFrame()
            params.update(
                {
                    "locations_id": ",".join(location_ids),
                    "parameters_id": "2",
                    "datetime_from": params.pop("date_from"),
                    "datetime_to": params.pop("date_to"),
                }
            )
        else:
            params.update({"city": city, "parameter": "pm25"})

        response = requests.get(api_url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("results") or payload.get("data") or []
        frame = pd.DataFrame(records)
        if frame.empty:
            return frame
        if "date" in frame.columns and frame["date"].map(lambda value: isinstance(value, dict)).any():
            timestamps = frame["date"].map(lambda value: value.get("utc") if isinstance(value, dict) else value)
        else:
            timestamp_column = next((col for col in ["date.utc", "date", "datetime", "timestamp"] if col in frame.columns), None)
            timestamps = frame[timestamp_column] if timestamp_column else None
        value_column = next((col for col in ["value", "pm25", "measurement"] if col in frame.columns), None)
        if timestamps is None or value_column is None:
            return pd.DataFrame()
        timestamps = pd.to_datetime(timestamps, utc=True, errors="coerce")
        pm25 = pd.to_numeric(frame[value_column], errors="coerce")
        station_column = next((col for col in ["location", "location_id", "sensor"] if col in frame.columns), None)
        output = pd.DataFrame(
            {
                "timestamp": timestamps,
                "city": city,
                "pm25_ugm3": pm25,
                "aqi": pm25.apply(lambda value: _pm25_to_aqi(float(value)) if pd.notna(value) else np.nan),
                "station_id": frame[station_column] if station_column else [get_city_config(city)["openaq_location"]] * len(frame),
                "source": SOURCE_NAME_OPENAQ,
            }
        )
        return output.dropna(subset=["timestamp", "aqi"])

    return _retry_call(_request)


def _fetch_openweather_weather(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    city_config = get_city_config(city)
    api_url = os.getenv(OPENWEATHER_API_URL_ENV, DEFAULT_OPENWEATHER_API_URL)
    api_key = os.getenv(OPENWEATHER_API_KEY_ENV, "").strip()
    if not api_key:
        return pd.DataFrame()

    start_day = start_date.floor("D")
    end_day = end_date.ceil("D")
    days = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    rows: list[pd.DataFrame] = []

    for day in days:
        params = {
            "lat": city_config["lat"],
            "lon": city_config["lon"],
            "dt": int(day.replace(hour=12).timestamp()),
            "appid": api_key,
            "units": "metric",
        }

        def _request_day() -> pd.DataFrame:
            response = requests.get(api_url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            records = payload.get("data") or payload.get("hourly") or []
            frame = pd.DataFrame(records)
            if frame.empty:
                return frame
            timestamp_column = next((col for col in ["dt", "timestamp", "time", "datetime"] if col in frame.columns), None)
            if timestamp_column is None:
                return pd.DataFrame()
            temp_column = next((col for col in ["temp", "temperature", "temp_c"] if col in frame.columns), None)
            humidity_column = next((col for col in ["humidity", "humidity_pct"] if col in frame.columns), None)
            wind_column = next((col for col in ["wind_speed", "wind"] if col in frame.columns), None)
            rain_column = next((col for col in ["rain", "precipitation"] if col in frame.columns), None)
            output = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(frame[timestamp_column], unit="s", utc=True, errors="coerce")
                    if pd.api.types.is_numeric_dtype(frame[timestamp_column])
                    else pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce"),
                    "city": city,
                    "temperature_c": pd.to_numeric(frame[temp_column], errors="coerce") if temp_column else np.nan,
                    "humidity_pct": pd.to_numeric(frame[humidity_column], errors="coerce") if humidity_column else np.nan,
                    "wind_speed_mps": pd.to_numeric(frame[wind_column], errors="coerce") if wind_column else np.nan,
                    "precipitation_mm": pd.to_numeric(frame[rain_column], errors="coerce") if rain_column else 0.0,
                    "source": SOURCE_NAME_OPENWEATHER,
                }
            )
            return output.dropna(subset=["timestamp"])

        day_frame = _retry_call(_request_day)
        if not day_frame.empty:
            rows.append(day_frame)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["timestamp", "city"])


def _synthetic_aqi_frame(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    synthetic = generate_synthetic_data(city, start_date, end_date)
    return synthetic[["timestamp", "city", "station_id", "aqi", "pm25_ugm3", "source"]].copy()


def _synthetic_weather_frame(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    synthetic = generate_synthetic_data(city, start_date, end_date)
    return synthetic[["timestamp", "city", "temperature_c", "humidity_pct", "wind_speed_mps", "precipitation_mm", "source"]].copy()


def fetch_aqi_data(city: str, start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
    """Fetch AQI data for a city, falling back to OpenAQ or synthetic data."""
    start_ts = _coerce_datetime(start_date)
    end_ts = _coerce_datetime(end_date)

    data = _retry_call(lambda: _fetch_cpcb_aqi(city, start_ts, end_ts))
    if data.empty:
        data = _retry_call(lambda: _fetch_openaq_aqi(city, start_ts, end_ts))
    if data.empty:
        data = _synthetic_aqi_frame(city, start_ts, end_ts)
    source_name = data["source"].iloc[0] if not data.empty and "source" in data.columns else SOURCE_NAME_SYNTHETIC
    _save_raw_pull(data, city, source_name, start_ts, end_ts)
    return data.sort_values("timestamp").reset_index(drop=True)


def fetch_weather_data(city: str, start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
    """Fetch weather data for a city, falling back to synthetic data when needed."""
    start_ts = _coerce_datetime(start_date)
    end_ts = _coerce_datetime(end_date)

    data = _retry_call(lambda: _fetch_openweather_weather(city, start_ts, end_ts))
    if data.empty:
        data = _synthetic_weather_frame(city, start_ts, end_ts)
    source_name = data["source"].iloc[0] if not data.empty and "source" in data.columns else SOURCE_NAME_SYNTHETIC
    _save_raw_pull(data, city, source_name, start_ts, end_ts)
    return data.sort_values("timestamp").reset_index(drop=True)
