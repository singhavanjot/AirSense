"""Test compatibility shims for the pandas version in this environment."""

from __future__ import annotations

import pandas as pd


_original_date_range = pd.date_range


def _normalize_frequency(value):
    if isinstance(value, str):
        return value.replace("H", "h").replace("T", "min")
    return value


def _compat_date_range(*args, **kwargs):
    if "freq" in kwargs:
        kwargs["freq"] = _normalize_frequency(kwargs["freq"])
    if len(args) >= 3 and isinstance(args[2], str):
        args = list(args)
        args[2] = _normalize_frequency(args[2])
    return _original_date_range(*args, **kwargs)


pd.date_range = _compat_date_range
