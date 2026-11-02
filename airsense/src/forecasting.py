"""Forecasting utilities for AirSense using Prophet when available."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import MODELS_DIR

try:  # pragma: no cover - optional dependency fallback
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency fallback
    Prophet = None


@dataclass
class ForecastEvaluation:
    """Time-aware holdout metrics for a trained model."""

    mae: float
    rmse: float
    train_size: int
    test_size: int
    model_type: str


class SimpleProphetLikeModel:
    """Fallback forecaster that mimics the small Prophet surface we use."""

    def __init__(self, interval_width: float = 0.8) -> None:
        self.interval_width = interval_width
        self.history_: pd.DataFrame | None = None
        self._baseline_mean: float = 0.0
        self._linear_coef: tuple[float, float] = (0.0, 0.0)
        self._hourly_profile: pd.Series | None = None
        self._weekday_profile: pd.Series | None = None
        self.evaluation_: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "SimpleProphetLikeModel":
        history = frame.copy().sort_values("ds").reset_index(drop=True)
        history["ds"] = pd.to_datetime(history["ds"], utc=True, errors="coerce").dt.tz_localize(None)
        history["y"] = pd.to_numeric(history["y"], errors="coerce")
        history = history.dropna(subset=["ds", "y"])
        self.history_ = history
        self._baseline_mean = float(history["y"].mean()) if not history.empty else 0.0
        if len(history) >= 2:
            x = np.arange(len(history), dtype=float)
            slope, intercept = np.polyfit(x, history["y"].to_numpy(), 1)
            self._linear_coef = (float(slope), float(intercept))
        else:
            self._linear_coef = (0.0, self._baseline_mean)

        self._hourly_profile = history.groupby(history["ds"].dt.hour)["y"].mean()
        self._weekday_profile = history.groupby(history["ds"].dt.dayofweek)["y"].mean()
        return self

    def make_future_dataframe(self, periods: int, freq: str = "h", include_history: bool = True) -> pd.DataFrame:
        if self.history_ is None or self.history_.empty:
            raise ValueError("Model has not been fitted.")
        history_dates = self.history_["ds"].copy()
        future_dates = pd.date_range(history_dates.iloc[-1] + pd.Timedelta(freq), periods=periods, freq=freq, tz="UTC")
        dates = pd.concat([history_dates, pd.Series(future_dates)], ignore_index=True) if include_history else pd.Series(future_dates)
        return pd.DataFrame({"ds": pd.to_datetime(dates, utc=True)})

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.history_ is None or self.history_.empty:
            raise ValueError("Model has not been fitted.")
        predictions = future.copy()
        predictions["ds"] = pd.to_datetime(predictions["ds"], utc=True)
        x = np.arange(len(predictions), dtype=float)
        slope, intercept = self._linear_coef
        trend = slope * x + intercept
        hourly = predictions["ds"].dt.hour.map(self._hourly_profile).fillna(self._baseline_mean).to_numpy()
        weekday = predictions["ds"].dt.dayofweek.map(self._weekday_profile).fillna(self._baseline_mean).to_numpy()
        yhat = 0.45 * trend + 0.35 * hourly + 0.20 * weekday
        residual_std = float(self.history_["y"].std(ddof=1) or 1.0)
        spread = max(residual_std * 1.2, 10.0)
        predictions["yhat"] = yhat
        predictions["yhat_lower"] = yhat - spread
        predictions["yhat_upper"] = yhat + spread
        return predictions


def _model_path(city: str) -> Path:
    safe_city = "".join(char.lower() if char.isalnum() else "_" for char in city).strip("_")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"{safe_city}_prophet.pkl"


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["timestamp", "aqi"]).sort_values("timestamp")
    return frame


def _make_model() -> Any:
    if Prophet is None:
        return SimpleProphetLikeModel(interval_width=0.8)
    return Prophet(interval_width=0.8, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)


def _to_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ds": pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None),
        "y": pd.to_numeric(df["aqi"], errors="coerce"),
    })


def _evaluate_holdout(model: Any, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict[str, Any]:
    if test_frame.empty:
        return {"mae": None, "rmse": None, "train_size": len(train_frame), "test_size": 0}

    if hasattr(model, "make_future_dataframe"):
        future = model.make_future_dataframe(periods=len(test_frame), freq="h", include_history=True)
        forecast = model.predict(future)
        forecast = forecast[["ds", "yhat"]].tail(len(test_frame)).reset_index(drop=True)
    else:  # pragma: no cover - defensive branch
        forecast = pd.DataFrame({"ds": test_frame["ds"], "yhat": np.repeat(train_frame["y"].mean(), len(test_frame))})

    actual = test_frame["y"].reset_index(drop=True)
    predicted = forecast["yhat"].reset_index(drop=True)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return {"mae": mae, "rmse": rmse, "train_size": len(train_frame), "test_size": len(test_frame)}


def train_forecast_model(df: pd.DataFrame, city: str) -> Any:
    """Train a Prophet-style forecast model on cleaned AQI data."""
    frame = _prepare_frame(df)
    prophet_frame = _to_prophet_frame(frame)
    if len(prophet_frame) < 8:
        model = _make_model()
        model.fit(prophet_frame)
        model.evaluation_ = {"mae": None, "rmse": None, "train_size": len(prophet_frame), "test_size": 0, "model_type": type(model).__name__}
        _store_model(model, city)
        return model

    holdout = max(6, min(24, max(1, len(prophet_frame) // 5)))
    train_frame = prophet_frame.iloc[:-holdout].copy()
    test_frame = prophet_frame.iloc[-holdout:].copy()

    model = _make_model()
    model.fit(train_frame)
    model.evaluation_ = _evaluate_holdout(model, train_frame, test_frame)
    model.evaluation_["model_type"] = type(model).__name__
    _store_model(model, city)
    return model


def _store_model(model: Any, city: str) -> None:
    path = _model_path(city)
    try:
        with path.open("wb") as handle:
            pickle.dump(model, handle)
    except Exception:  # pragma: no cover - best effort caching
        return


def load_cached_model(city: str) -> Any | None:
    """Load a cached forecast model if one exists."""
    path = _model_path(city)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:  # pragma: no cover - cache corruption fallback
        return None


def load_or_train_model(df: pd.DataFrame, city: str) -> Any:
    """Load a cached model or train a fresh one when needed."""
    cached = load_cached_model(city)
    if cached is not None:
        return cached
    return train_forecast_model(df, city)


def predict_next_hours(model: Any, hours: int = 24) -> pd.DataFrame:
    """Predict the next N hours using a trained model."""
    if hours < 1:
        raise ValueError("hours must be positive")
    future = model.make_future_dataframe(periods=hours, freq="h", include_history=True)
    forecast = model.predict(future).tail(hours).copy()
    if "yhat" not in forecast.columns:
        forecast = forecast.rename(columns={"forecast": "yhat"})
    output = forecast.rename(columns={"ds": "timestamp", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"})
    return output[["timestamp", "forecast", "lower", "upper"]].reset_index(drop=True)
