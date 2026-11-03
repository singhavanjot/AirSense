from __future__ import annotations

import pandas as pd

from airsense_src.correlation_analysis import compute_peak_offpeak_stats, partial_correlation_controlling_weather


def _sample_frame(all_peak: bool = False) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=24, freq="H", tz="UTC")
    is_peak = [True] * 24 if all_peak else [i % 2 == 0 for i in range(24)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": ["Amritsar"] * 24,
            "aqi": [120 + (10 if peak else -5) for peak in is_peak],
            "temperature_c": [20.0 + (i % 4) for i in range(24)],
            "humidity_pct": [55.0 + (i % 5) for i in range(24)],
            "wind_speed_mps": [2.0 + (i % 3) * 0.2 for i in range(24)],
            "is_peak_hour": is_peak,
        }
    )


def test_compute_peak_offpeak_stats_handles_all_peak_data() -> None:
    result = compute_peak_offpeak_stats(_sample_frame(all_peak=True))

    assert result["test_used"] == "insufficient_data"
    assert "interpretation" in result
    assert result["p_value"] is None


def test_partial_correlation_returns_expected_structure() -> None:
    result = partial_correlation_controlling_weather(_sample_frame(all_peak=False))

    assert set(["correlation", "p_value", "n", "covariates", "method", "interpretation"]).issubset(result.keys())
    assert result["method"] == "residual_regression"
    assert isinstance(result["covariates"], list)
