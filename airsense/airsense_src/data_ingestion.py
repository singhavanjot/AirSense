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

from airsense_config import (
    CPCB_API_KEY_ENV,
    CPCB_API_URL_ENV,
    DEFAULT_CPCB_API_URL,
    DEFAULT_OPENAQ_API_URL,
    DEFAULT_OPENWEATHER_API_URL,
    DEFAULT_OPEN_METEO_API_URL,
    OPENAQ_API_KEY_ENV,
    OPENAQ_API_URL_ENV,
    OPENWEATHER_API_KEY_ENV,
    OPENWEATHER_API_URL_ENV,
    OPEN_METEO_API_URL_ENV,
    RAW_DATA_DIR,
    RANDOM_SEED,
    SOURCE_NAME_CPCB,
    SOURCE_NAME_OPENAQ,
    SOURCE_NAME_OPENWEATHER,
    SOURCE_NAME_OPEN_METEO,
    SOURCE_NAME_SYNTHETIC,
    get_city_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostic log – collects human-readable notes about each API attempt
# so the dashboard can display them.  Reset per session.
# ---------------------------------------------------------------------------
_api_diagnostics: list[str] = []


def get_api_diagnostics() -> list[str]:
    """Return accumulated diagnostic messages for the current session."""
    return list(_api_diagnostics)


def clear_api_diagnostics() -> None:
    """Reset diagnostic messages (called at the start of each dashboard load)."""
    _api_diagnostics.clear()


def _diag(message: str) -> None:
    """Append a diagnostic line *and* emit it via the logger."""
    _api_diagnostics.append(message)
    logger.info(message)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_error_message(error: Exception) -> str:
    """Return an exception message with common API credential query values hidden."""
    message = str(error)
    for parameter in ("appid", "api-key", "X-API-Key", "apikey", "key"):
        marker = f"{parameter}="
        while marker in message:
            start = message.index(marker) + len(marker)
            end = message.find("&", start)
            if end == -1:
                end = len(message)
            message = f"{message[:start]}[redacted]{message[end:]}"
    return message


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find a response column while tolerating case, spaces, and punctuation differences."""
    normalized = {
        "".join(character for character in str(column).lower() if character.isalnum()): column
        for column in frame.columns
    }
    for candidate in candidates:
        key = "".join(character for character in candidate.lower() if character.isalnum())
        if key in normalized:
            return normalized[key]
    return None


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
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code in {401, 403}:
                _diag(f"  ✗ HTTP {status_code} – authentication failed")
                return pd.DataFrame()
            wait_time = base_delay * (2**attempt)
            logger.warning("Attempt %s failed: %s", attempt + 1, _safe_error_message(exc))
            if attempt < attempts - 1:
                time.sleep(wait_time)
    if last_error is not None:
        _diag(f"  ✗ All retries failed: {_safe_error_message(last_error)}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CPCB (data.gov.in) AQI ingestion – uses filters[] syntax
# ---------------------------------------------------------------------------

def _fetch_cpcb_aqi(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    api_url = os.getenv(CPCB_API_URL_ENV, DEFAULT_CPCB_API_URL)
    api_key = os.getenv(CPCB_API_KEY_ENV, "").strip()
    if not api_key:
        _diag("CPCB: skipped – no API key set (register at data.gov.in)")
        return pd.DataFrame()

    _diag(f"CPCB: fetching for {city} from data.gov.in…")
    # data.gov.in uses filters[column]=value syntax
    params: dict[str, Any] = {
        "api-key": api_key,
        "format": "json",
        "limit": 1000,
        "offset": 0,
        "filters[city]": city,
    }

    def _request() -> pd.DataFrame:
        response = requests.get(api_url, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("records") or payload.get("data") or payload.get("results") or []
        frame = pd.DataFrame(records)

        # If empty with exact city, try title case or uppercase variants
        if frame.empty:
            for variant in [city.title(), city.upper()]:
                params["filters[city]"] = variant
                r2 = requests.get(api_url, params=params, timeout=25)
                if r2.status_code == 200:
                    rec2 = r2.json().get("records") or []
                    if rec2:
                        frame = pd.DataFrame(rec2)
                        break

        if frame.empty:
            _diag(f"  ✗ CPCB returned 0 records for {city}")
            return frame

        # CPCB returns rows per pollutant (PM2.5, PM10, NO2, etc.)
        # Prioritize PM2.5 if available
        if "pollutant_id" in frame.columns:
            pm25_mask = frame["pollutant_id"].astype(str).str.upper().str.contains("PM2.5|PM25|PM2_5")
            if pm25_mask.any():
                frame = frame[pm25_mask]

        timestamp_column = _find_column(frame, ("timestamp", "date", "datetime", "recorded_at", "last_update"))
        aqi_column = _find_column(
            frame,
            ("aqi", "air_quality_index", "aqi_value", "aqi_value_numeric", "avg_aqi", "pollution_index", "value", "pollutant_avg"),
        )
        if timestamp_column is None or aqi_column is None:
            _diag(f"  ✗ CPCB columns not found (got: {list(frame.columns)[:10]})")
            return pd.DataFrame()

        raw_ts = frame[timestamp_column]
        parsed_ts = pd.to_datetime(raw_ts, dayfirst=True, utc=True, errors="coerce")
        raw_val = pd.to_numeric(frame[aqi_column], errors="coerce")

        if aqi_column == "pollutant_avg":
            aqi_series = raw_val.apply(lambda v: _pm25_to_aqi(float(v)) if pd.notna(v) else np.nan)
            pm25_series = raw_val
        else:
            aqi_series = raw_val
            pm25_series = aqi_series.apply(lambda v: max(0.0, (float(v) - 10.0) / 4.0) if pd.notna(v) else np.nan)

        station_col = _find_column(frame, ("station", "station_id", "station_name", "location"))
        output = pd.DataFrame(
            {
                "timestamp": parsed_ts,
                "city": city,
                "aqi": aqi_series,
                "pm25_ugm3": pm25_series,
                "station_id": frame[station_col] if station_col else [get_city_config(city)["cpcb_station_id"]] * len(frame),
                "source": SOURCE_NAME_CPCB,
            }
        )
        result = output.dropna(subset=["timestamp", "aqi"])
        _diag(f"  ✓ CPCB returned {len(result)} records for {city}")
        return result

    return _retry_call(_request)


# ---------------------------------------------------------------------------
# OpenAQ v3 AQI ingestion – coordinate-based sensor lookup
# ---------------------------------------------------------------------------

def _fetch_openaq_aqi(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    api_base = os.getenv(OPENAQ_API_URL_ENV, DEFAULT_OPENAQ_API_URL).rstrip("/")
    api_key = os.getenv(OPENAQ_API_KEY_ENV, "").strip()

    # OpenAQ v3 requires an API key – skip if missing
    if not api_key:
        _diag("OpenAQ: skipped – no API key set")
        return pd.DataFrame()

    headers = {"accept": "application/json", "X-API-Key": api_key}
    city_cfg = get_city_config(city)
    _diag(f"OpenAQ v3: fetching for {city}…")

    def _request() -> pd.DataFrame:
        # Step 1: Find locations near the city coordinates
        locations_url = f"{api_base}/locations"
        loc_params = {
            "coordinates": f"{city_cfg['lat']},{city_cfg['lon']}",
            "radius": 25000,
            "limit": 10,
        }
        loc_response = requests.get(locations_url, params=loc_params, headers=headers, timeout=20)
        loc_response.raise_for_status()
        locations = loc_response.json().get("results") or []
        if not locations:
            _diag(f"  ✗ OpenAQ found no locations near {city}")
            return pd.DataFrame()

        _diag(f"  → Found {len(locations)} location(s)")

        # Step 2: Collect sensor IDs for PM2.5 (parameters_id=2) from each location
        all_rows: list[pd.DataFrame] = []
        for loc in locations[:3]:  # limit to 3 locations to avoid rate limits
            location_id = loc.get("id")
            if location_id is None:
                continue

            # Get sensors for this location
            sensors_url = f"{api_base}/locations/{location_id}/sensors"
            sensors_response = requests.get(sensors_url, headers=headers, timeout=20)
            if sensors_response.status_code != 200:
                continue
            sensors = sensors_response.json().get("results") or []

            for sensor in sensors:
                sensor_id = sensor.get("id")
                param = sensor.get("parameter") or {}
                param_id = param.get("id") if isinstance(param, dict) else None
                # PM2.5 has parameters_id = 2 in OpenAQ
                if param_id != 2 and str(param_id) != "2":
                    continue

                # Step 3: Get hourly measurements from this sensor
                measurements_url = f"{api_base}/sensors/{sensor_id}/hours"
                meas_params = {
                    "datetime_from": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "datetime_to": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 1000,
                }
                meas_response = requests.get(measurements_url, params=meas_params, headers=headers, timeout=20)
                if meas_response.status_code != 200:
                    continue
                records = meas_response.json().get("results") or []
                if not records:
                    continue

                frame = pd.DataFrame(records)
                # OpenAQ v3 hourly response has 'datetime' (dict with 'utc') and 'value'
                if "datetime" in frame.columns:
                    timestamps = frame["datetime"].apply(
                        lambda d: d.get("utc") if isinstance(d, dict) else d
                    )
                elif "period" in frame.columns:
                    timestamps = frame["period"].apply(
                        lambda d: d.get("datetimeFrom", {}).get("utc") if isinstance(d, dict) else d
                    )
                else:
                    ts_col = next((c for c in ["date", "timestamp"] if c in frame.columns), None)
                    if ts_col is None:
                        continue
                    timestamps = frame[ts_col]

                value_col = next((c for c in ["value", "average", "avg"] if c in frame.columns), None)
                if value_col is None:
                    continue

                pm25 = pd.to_numeric(frame[value_col], errors="coerce")
                sensor_df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
                    "city": city,
                    "pm25_ugm3": pm25,
                    "aqi": pm25.apply(lambda v: _pm25_to_aqi(float(v)) if pd.notna(v) else np.nan),
                    "station_id": str(location_id),
                    "source": SOURCE_NAME_OPENAQ,
                })
                all_rows.append(sensor_df.dropna(subset=["timestamp", "aqi"]))

        if not all_rows:
            _diag("  ✗ OpenAQ returned no PM2.5 measurements")
            return pd.DataFrame()

        combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["timestamp", "city"])
        _diag(f"  ✓ OpenAQ returned {len(combined)} records")
        return combined

    return _retry_call(_request)


# ---------------------------------------------------------------------------
# OpenWeatherMap weather ingestion
# ---------------------------------------------------------------------------

def _fetch_openweather_weather(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    city_config = get_city_config(city)
    api_url = os.getenv(OPENWEATHER_API_URL_ENV, DEFAULT_OPENWEATHER_API_URL)
    api_key = os.getenv(OPENWEATHER_API_KEY_ENV, "").strip()
    if not api_key:
        _diag("OpenWeather: skipped – no API key set")
        return pd.DataFrame()

    _diag(f"OpenWeather: fetching for {city}…")
    start_day = start_date.floor("D")
    end_day = end_date.ceil("D")
    days = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    rows: list[pd.DataFrame] = []
    authorization_failed = False

    for day in days:
        params = {
            "lat": city_config["lat"],
            "lon": city_config["lon"],
            "dt": int(day.replace(hour=12).timestamp()),
            "appid": api_key,
            "units": "metric",
        }

        def _request_day() -> pd.DataFrame:
            nonlocal authorization_failed
            response = requests.get(api_url, params=params, timeout=20)
            if response.status_code in {401, 403}:
                authorization_failed = True
                _diag(f"  ✗ OpenWeather HTTP {response.status_code}")
                return pd.DataFrame()
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

            # Handle rain field – OpenWeather returns {"1h": value} dicts
            rain_values: Any = 0.0
            if rain_column and rain_column in frame.columns:
                rain_values = frame[rain_column].apply(
                    lambda v: v.get("1h", 0.0) if isinstance(v, dict) else (float(v) if pd.notna(v) else 0.0)
                )

            output = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(frame[timestamp_column], unit="s", utc=True, errors="coerce")
                    if pd.api.types.is_numeric_dtype(frame[timestamp_column])
                    else pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce"),
                    "city": city,
                    "temperature_c": pd.to_numeric(frame[temp_column], errors="coerce") if temp_column else np.nan,
                    "humidity_pct": pd.to_numeric(frame[humidity_column], errors="coerce") if humidity_column else np.nan,
                    "wind_speed_mps": pd.to_numeric(frame[wind_column], errors="coerce") if wind_column else np.nan,
                    "precipitation_mm": rain_values,
                    "source": SOURCE_NAME_OPENWEATHER,
                }
            )
            return output.dropna(subset=["timestamp"])

        day_frame = _retry_call(_request_day)
        if authorization_failed:
            break
        if not day_frame.empty:
            rows.append(day_frame)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["timestamp", "city"])
    _diag(f"  ✓ OpenWeather returned {len(result)} records")
    return result


# ---------------------------------------------------------------------------
# Open-Meteo FREE weather fallback – no API key required, hourly resolution
# ---------------------------------------------------------------------------

def _fetch_open_meteo_weather(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo (free, no API key)."""
    city_config = get_city_config(city)
    api_url = os.getenv(OPEN_METEO_API_URL_ENV, DEFAULT_OPEN_METEO_API_URL)

    _diag(f"Open-Meteo: fetching for {city}…")

    # Clamp end_date to yesterday (Open-Meteo archive has ~5-day lag for ERA5)
    today = pd.Timestamp.now(tz="UTC").floor("D")
    effective_end = min(end_date.floor("D"), today - pd.Timedelta(days=1))
    effective_start = start_date.floor("D")
    if effective_end < effective_start:
        # If the entire range is too recent, shift start back to get *some* data
        effective_end = today - pd.Timedelta(days=1)
        effective_start = effective_end - pd.Timedelta(days=max(7, (end_date - start_date).days))

    params = {
        "latitude": city_config["lat"],
        "longitude": city_config["lon"],
        "start_date": effective_start.strftime("%Y-%m-%d"),
        "end_date": effective_end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    def _request() -> pd.DataFrame:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            _diag("  ✗ Open-Meteo returned no hourly data")
            return pd.DataFrame()

        output = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"], utc=True, errors="coerce"),
            "city": city,
            "temperature_c": pd.to_numeric(pd.Series(hourly.get("temperature_2m")), errors="coerce"),
            "humidity_pct": pd.to_numeric(pd.Series(hourly.get("relative_humidity_2m")), errors="coerce"),
            "wind_speed_mps": pd.to_numeric(pd.Series(hourly.get("wind_speed_10m")), errors="coerce"),
            "precipitation_mm": pd.to_numeric(pd.Series(hourly.get("precipitation")), errors="coerce").fillna(0.0),
            "source": SOURCE_NAME_OPEN_METEO,
        })
        result = output.dropna(subset=["timestamp"])
        _diag(f"  ✓ Open-Meteo returned {len(result)} records")
        return result

    return _retry_call(_request)


# ---------------------------------------------------------------------------
# Synthetic fallbacks
# ---------------------------------------------------------------------------

def _synthetic_aqi_frame(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    synthetic = generate_synthetic_data(city, start_date, end_date)
    return synthetic[["timestamp", "city", "station_id", "aqi", "pm25_ugm3", "source"]].copy()


def _synthetic_weather_frame(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    synthetic = generate_synthetic_data(city, start_date, end_date)
    return synthetic[["timestamp", "city", "temperature_c", "humidity_pct", "wind_speed_mps", "precipitation_mm", "source"]].copy()


# ---------------------------------------------------------------------------
# Public API – fetch_aqi_data / fetch_weather_data
# ---------------------------------------------------------------------------

def fetch_aqi_data(city: str, start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
    """Fetch AQI data for a city, falling back to OpenAQ or synthetic data."""
    start_ts = _coerce_datetime(start_date)
    end_ts = _coerce_datetime(end_date)

    data = _retry_call(lambda: _fetch_cpcb_aqi(city, start_ts, end_ts))
    if data.empty:
        data = _retry_call(lambda: _fetch_openaq_aqi(city, start_ts, end_ts))
    if data.empty:
        _diag(f"AQI [{city}]: all APIs failed → synthetic fallback")
        data = _synthetic_aqi_frame(city, start_ts, end_ts)
    source_name = data["source"].iloc[0] if not data.empty and "source" in data.columns else SOURCE_NAME_SYNTHETIC
    _save_raw_pull(data, city, source_name, start_ts, end_ts)
    return data.sort_values("timestamp").reset_index(drop=True)


def fetch_weather_data(city: str, start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
    """Fetch weather data for a city, falling back through Open-Meteo and synthetic data."""
    start_ts = _coerce_datetime(start_date)
    end_ts = _coerce_datetime(end_date)

    # Try OpenWeather first (requires paid API key)
    data = _retry_call(lambda: _fetch_openweather_weather(city, start_ts, end_ts))

    # Fallback to Open-Meteo (free, no API key needed)
    if data.empty:
        data = _retry_call(lambda: _fetch_open_meteo_weather(city, start_ts, end_ts))

    # Last resort: synthetic
    if data.empty:
        _diag(f"Weather [{city}]: all APIs failed → synthetic fallback")
        data = _synthetic_weather_frame(city, start_ts, end_ts)

    source_name = data["source"].iloc[0] if not data.empty and "source" in data.columns else SOURCE_NAME_SYNTHETIC
    _save_raw_pull(data, city, source_name, start_ts, end_ts)
    return data.sort_values("timestamp").reset_index(drop=True)
