"""AQI health advisory mapping for AirSense."""
from __future__ import annotations

from config import AQI_THRESHOLDS, CATEGORY_COLORS, HEALTH_ACTIONS


def aqi_to_category(aqi_value: float | int | None) -> str:
    """Map an AQI value to the CPCB category bands."""
    if aqi_value is None:
        return "Unknown"

    try:
        aqi = float(aqi_value)
    except (TypeError, ValueError):
        return "Unknown"

    for category, (lower, upper) in AQI_THRESHOLDS.items():
        if lower <= aqi <= upper:
            return category
    if aqi < 0:
        return "Unknown"
    return "Severe"


def get_recommended_action(category: str) -> str:
    """Return a human-readable recommendation for the AQI category."""
    return HEALTH_ACTIONS.get(category, "No advisory available for this category.")


def get_category_color(category: str) -> str:
    """Return a display color for the AQI category."""
    return CATEGORY_COLORS.get(category, "#607D8B")
