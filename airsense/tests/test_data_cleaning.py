from __future__ import annotations

import pandas as pd

from airsense_src.data_cleaning import clean_aqi_data, tag_peak_hours


def test_clean_aqi_data_removes_invalid_values_and_fills_missing() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00", periods=5, freq="H", tz="UTC"),
            "city": ["Ludhiana"] * 5,
            "aqi": [50, None, -3, 999, 55],
            "temperature_c": [20.0, None, 21.0, 22.0, 23.0],
            "humidity_pct": [55.0, 56.0, None, 58.0, 59.0],
        }
    )

    cleaned = clean_aqi_data(frame)

    assert not cleaned.empty
    assert cleaned["aqi"].between(0, 500).all()
    assert cleaned["aqi"].isna().sum() == 0
    assert cleaned["temperature_c"].isna().sum() == 0
    assert cleaned["humidity_pct"].isna().sum() == 0


def test_tag_peak_hours_sets_expected_flags() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05 08:00", "2026-01-05 14:00", "2026-01-05 18:00"], utc=True),
            "city": ["Chandigarh"] * 3,
            "aqi": [110, 90, 120],
        }
    )

    tagged = tag_peak_hours(frame)

    assert tagged["is_peak_hour"].tolist() == [True, False, True]
    assert tagged["peak_period"].tolist() == ["peak", "off_peak", "peak"]
