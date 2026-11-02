# AirSense

AirSense is a batch-processed AQI intelligence project for Punjab cities. It ingests AQI and weather data, cleans and merges the records, analyzes peak-traffic-hour effects while controlling for weather, forecasts short-horizon AQI, and presents findings in a Streamlit dashboard with health advisories and downloadable reports.

## Setup

1. Create and activate a virtual environment if needed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file and set any real API keys you want to use:

```bash
cp .env.example .env
```

The app works with synthetic data even when no API keys are configured, so a first run can succeed without external access.

## Run the dashboard

```bash
streamlit run app.py
```

## Run tests

```bash
pytest
```

## Architecture

```mermaid
flowchart LR
    A[Data Sources<br/>CPCB / OpenAQ / OpenWeatherMap] --> B[Ingestion & Cleaning]
    B --> C[Correlation + Forecasting]
    C --> D[Health Advisory + Alerts]
    D --> E[Dashboard]
```

## Known limitations

- Traffic impact is represented using time-window proxies rather than live GPS or road-level traffic measurements.
- Forecasting is intentionally capped to a 24-hour horizon because the project is designed as a short-term, batch-oriented dashboard rather than a real-time forecasting system.
- Some live API integrations are optional; synthetic data is used automatically when keys or rate limits are unavailable.

## Repository layout

```text
airsense/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
├── data/
├── tests/
├── notebooks/
└── .venv/
```
