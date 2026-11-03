from __future__ import annotations

import pytest

from airsense_src.health_advisory import aqi_to_category


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "Good"),
        (50, "Good"),
        (51, "Satisfactory"),
        (100, "Satisfactory"),
        (101, "Moderate"),
        (200, "Moderate"),
        (201, "Poor"),
        (300, "Poor"),
        (301, "Very Poor"),
        (400, "Very Poor"),
        (401, "Severe"),
        (500, "Severe"),
    ],
)
def test_aqi_to_category_matches_cpcb_thresholds(value: int, expected: str) -> None:
    assert aqi_to_category(value) == expected
