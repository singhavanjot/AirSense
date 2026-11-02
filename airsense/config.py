"""Central configuration for the AirSense project."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROCESSED_DATA_DIR / "models"
DATABASE_PATH = DATA_DIR / "airsense.sqlite3"

for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CITIES = {
    "Ludhiana": {
        "lat": 30.900965,
        "lon": 75.857275,
        "cpcb_station_id": "Ludhiana - Punjab PCB",
        "openaq_location": "Ludhiana",
    },
    "Chandigarh": {
        "lat": 30.7333148,
        "lon": 76.7794179,
        "cpcb_station_id": "Sector 22, Chandigarh",
        "openaq_location": "Chandigarh",
    },
    "Amritsar": {
        "lat": 31.6340,
        "lon": 74.8723,
        "cpcb_station_id": "Ranjit Avenue, Amritsar",
        "openaq_location": "Amritsar",
    },
}

PEAK_HOUR_WINDOWS = {
    "weekdays": [0, 1, 2, 3, 4],
    "morning": (8, 10),
    "evening": (17, 20),
    "weekend": (10, 18),
}

AQI_THRESHOLDS = {
    "Good": (0, 50),
    "Satisfactory": (51, 100),
    "Moderate": (101, 200),
    "Poor": (201, 300),
    "Very Poor": (301, 400),
    "Severe": (401, 500),
}

CATEGORY_COLORS = {
    "Good": "#2E7D32",
    "Satisfactory": "#66BB6A",
    "Moderate": "#F9A825",
    "Poor": "#EF6C00",
    "Very Poor": "#C62828",
    "Severe": "#4A148C",
}

HEALTH_ACTIONS = {
    "Good": "Air quality is satisfactory. Routine outdoor activity is fine.",
    "Satisfactory": "Sensitive groups should consider reducing prolonged outdoor exertion.",
    "Moderate": "Sensitive people should limit prolonged outdoor activity and use precautions.",
    "Poor": "Reduce outdoor exertion, especially for children, older adults, and people with lung conditions.",
    "Very Poor": "Avoid outdoor activity when possible; consider masks and indoor air filtration.",
    "Severe": "Avoid outdoor exposure; everyone should minimize time outside and use respiratory protection.",
}

DEFAULT_HISTORY_DAYS = 30
DEFAULT_FORECAST_HOURS = 24
MIN_FORECAST_HOURS = 6
MAX_FORECAST_HOURS = 24
RANDOM_SEED = 42

CPCB_API_URL_ENV = "CPCB_API_URL"
CPCB_API_KEY_ENV = "CPCB_API_KEY"
OPENAQ_API_URL_ENV = "OPENAQ_API_URL"
OPENAQ_API_KEY_ENV = "OPENAQ_API_KEY"
OPENWEATHER_API_URL_ENV = "OPENWEATHER_API_URL"
OPENWEATHER_API_KEY_ENV = "OPENWEATHER_API_KEY"

DEFAULT_CPCB_API_URL = "https://api.data.gov.in/resource/placeholder-cpcb-aqi"
DEFAULT_OPENAQ_API_URL = "https://api.openaq.org/v2/measurements"
DEFAULT_OPENWEATHER_API_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

SOURCE_NAME_CPCB = "cpcb"
SOURCE_NAME_OPENAQ = "openaq"
SOURCE_NAME_OPENWEATHER = "openweather"
SOURCE_NAME_SYNTHETIC = "synthetic"

TIMEZONE = "Asia/Kolkata"


def get_city_config(city: str) -> dict:
    """Return city metadata for a supported city."""
    if city not in CITIES:
        raise KeyError(f"Unsupported city: {city}")
    return CITIES[city]
