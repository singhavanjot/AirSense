"""Report export helpers for AirSense."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.health_advisory import aqi_to_category, get_recommended_action


def _format_forecast_preview(forecast: pd.DataFrame) -> str:
    if forecast.empty:
        return "No forecast data available."
    preview = forecast.copy().head(12)
    preview["timestamp"] = pd.to_datetime(preview["timestamp"], utc=True, errors="coerce")
    return preview.to_string(index=False)


def _format_alert_preview(alerts: pd.DataFrame) -> str:
    if alerts.empty:
        return "No severe AQI hours flagged."
    preview = alerts.copy().head(12)
    preview["timestamp"] = pd.to_datetime(preview["timestamp"], utc=True, errors="coerce")
    return preview.to_string(index=False)


def generate_summary_report(city: str, stats: dict[str, Any], forecast: pd.DataFrame, alerts: pd.DataFrame) -> str:
    """Generate a downloadable plain-text summary report for a city."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    latest_aqi = None
    latest_category = "Unknown"
    latest_action = "No data available."
    if not forecast.empty and "forecast" in forecast.columns:
        latest_aqi = float(forecast["forecast"].iloc[0])
        latest_category = aqi_to_category(latest_aqi)
        latest_action = get_recommended_action(latest_category)

    lines = [
        f"AirSense Summary Report - {city}",
        f"Generated: {now}",
        "",
        "Key Findings",
        f"- Peak vs off-peak interpretation: {stats.get('interpretation', 'Unavailable')}",
        f"- Partial correlation: {stats.get('partial', {}).get('interpretation', 'Unavailable')}",
        f"- Severe-hour share: {stats.get('alerts', {}).get('flagged_share_pct', 0.0)}%",
        "",
        "Forecast Preview",
        _format_forecast_preview(forecast),
        "",
        "Alert Preview",
        _format_alert_preview(alerts),
        "",
        "Advisory",
        f"- Next-hour forecast category: {latest_category}",
        f"- Suggested action: {latest_action}",
    ]
    return "\n".join(lines)
