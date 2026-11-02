"""Streamlit dashboard for AirSense."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_FORECAST_HOURS, DEFAULT_HISTORY_DAYS, MAX_FORECAST_HOURS, MIN_FORECAST_HOURS, PEAK_HOUR_WINDOWS, CITIES
from src.alerts import flag_severe_hours, summarize_alerts
from src.correlation_analysis import compute_peak_offpeak_stats, partial_correlation_controlling_weather
from src.data_cleaning import clean_aqi_data, merge_aqi_weather, save_cleaned_output, tag_peak_hours
from src.data_ingestion import fetch_aqi_data, fetch_weather_data
from src.forecasting import load_or_train_model, predict_next_hours
from src.health_advisory import aqi_to_category, get_category_color, get_recommended_action
from src.report_export import generate_summary_report

load_dotenv()

st.set_page_config(page_title="AirSense", page_icon="🌫️", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #0B1320;
        --panel: #101A2C;
        --panel-2: #16233A;
        --text: #F4F7FB;
        --muted: #AAB7C7;
        --accent: #5BD6A2;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(91, 214, 162, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(97, 165, 255, 0.12), transparent 22%),
            linear-gradient(180deg, #08111E 0%, #0D1728 100%);
        color: var(--text);
    }
    .hero {
        background: linear-gradient(135deg, rgba(16, 26, 44, 0.96), rgba(22, 35, 58, 0.90));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 24px 60px rgba(0,0,0,0.22);
    }
    .metric-card {
        background: rgba(16, 26, 44, 0.96);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        min-height: 150px;
        box-shadow: 0 18px 36px rgba(0,0,0,0.18);
    }
    .metric-label { color: var(--muted); font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { font-size: 2rem; font-weight: 800; margin: 0.2rem 0 0.3rem; }
    .metric-subtext { color: var(--muted); font-size: 0.92rem; }
    .section-title { margin-top: 1rem; margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _batch_marker() -> str:
    """Return the current UTC batch label for the dashboard refresh window."""
    return pd.Timestamp.now(tz="UTC").floor("D").strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False)
def load_city_dataset(city: str, batch_marker: str, history_days: int) -> pd.DataFrame:
    """Load, clean, merge, and tag one city's AQI and weather data."""
    end_date = pd.Timestamp.now(tz="UTC").floor("h")
    start_date = end_date - pd.Timedelta(days=history_days)
    aqi_raw = fetch_aqi_data(city, start_date, end_date)
    weather_raw = fetch_weather_data(city, start_date, end_date)
    cleaned_aqi = clean_aqi_data(aqi_raw)
    merged = merge_aqi_weather(cleaned_aqi, weather_raw)
    tagged = tag_peak_hours(merged)
    save_cleaned_output(tagged, city)
    return tagged


@st.cache_resource(show_spinner=False)
def load_city_model(city: str, batch_marker: str, history_days: int):
    """Load a cached forecast model for the city or train one if missing."""
    data = load_city_dataset(city, batch_marker, history_days)
    return load_or_train_model(data, city)


def _latest_record(frame: pd.DataFrame) -> pd.Series:
    """Return the most recent row from a city dataset."""
    return frame.sort_values("timestamp").iloc[-1]


def _render_card(title: str, value: str, subtitle: str, color: str) -> None:
    """Render a styled metric card in the dashboard."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            <div class="metric-subtext">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _trend_chart(city_frames: dict[str, pd.DataFrame], primary_city: str) -> go.Figure:
    """Create the historical AQI trend plot with peak-hour overlays."""
    fig = go.Figure()
    palette = ["#5BD6A2", "#61A5FF", "#FFD166", "#FF7A59"]
    for index, (city, frame) in enumerate(city_frames.items()):
        color = palette[index % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["aqi"],
                mode="lines",
                name=city,
                line=dict(color=color, width=2.5),
            )
        )
        if city == primary_city:
            peak_points = frame[frame["is_peak_hour"]]
            fig.add_trace(
                go.Scatter(
                    x=peak_points["timestamp"],
                    y=peak_points["aqi"],
                    mode="markers",
                    name=f"{city} peak hours",
                    marker=dict(size=6, color="#FFCC80", symbol="circle"),
                    hovertemplate="%{x}<br>AQI: %{y:.0f}<extra>Peak hour</extra>",
                    showlegend=True,
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        legend_title_text="City",
        yaxis_title="AQI",
        xaxis_title="Timestamp (UTC)",
    )
    return fig


def _forecast_chart(forecast: pd.DataFrame) -> go.Figure:
    """Create the forecast chart with confidence intervals."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#5BD6A2", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["upper"],
            mode="lines",
            line=dict(width=0),
            name="Upper bound",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(91, 214, 162, 0.18)",
            line=dict(width=0),
            name="Confidence band",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        yaxis_title="Forecast AQI",
        xaxis_title="Timestamp (UTC)",
    )
    return fig


def _heatmap_figure(frame: pd.DataFrame) -> go.Figure:
    """Build the day-of-week versus hour-of-day AQI heatmap."""
    heatmap_source = frame.copy()
    heatmap_source["day_name"] = heatmap_source["timestamp"].dt.day_name()
    heatmap_source["hour"] = heatmap_source["timestamp"].dt.hour
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = heatmap_source.pivot_table(index="day_name", columns="hour", values="aqi", aggfunc="mean").reindex(ordered_days)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="AQI"),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis_title="Hour of day",
        yaxis_title="Day of week",
    )
    return fig


def _analysis_sentence(stats: dict, partial: dict) -> str:
    """Translate the correlation findings into a plain-language interpretation."""
    peak_mean = stats.get("peak", {}).get("mean")
    off_mean = stats.get("off_peak", {}).get("mean")
    p_value = stats.get("p_value")
    if peak_mean is None or off_mean is None or p_value is None:
        return "The comparison is inconclusive because one of the groups is too small or missing."
    if p_value < 0.05:
        direction = "higher" if peak_mean > off_mean else "lower"
        return f"Peak hours are {direction} than off-peak hours in this sample, and the statistical test suggests the difference is unlikely to be random."
    if partial.get("correlation") is not None and partial.get("p_value") is not None and partial["p_value"] < 0.05:
        return "Weather-adjusted analysis finds a modest peak-hour signal even though the raw difference is not strong."
    return "The raw and weather-adjusted signals are weak, so peak traffic windows alone do not explain most of the AQI variation here."


st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0; font-size: 2.2rem;">AirSense</h1>
        <p style="margin:0.4rem 0 0; color: #AAB7C7;">Batch-processed AQI intelligence for Punjab cities with forecast, advisory, and alert summaries.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("City comparison")
selected_cities = st.sidebar.multiselect("Choose cities", list(CITIES.keys()), default=list(CITIES.keys())[:3])
if not selected_cities:
    st.info("Choose one or more cities to load the dashboard.")
    st.stop()

primary_city = st.sidebar.selectbox("Primary city for deep analysis", selected_cities)
forecast_hours = st.sidebar.slider("Forecast horizon (hours)", MIN_FORECAST_HOURS, MAX_FORECAST_HOURS, DEFAULT_FORECAST_HOURS, step=6)
history_days = st.sidebar.slider("Historical window (days)", 7, 90, DEFAULT_HISTORY_DAYS, step=7)

batch_marker = _batch_marker()
city_frames: dict[str, pd.DataFrame] = {}
for city in selected_cities:
    city_frames[city] = load_city_dataset(city, batch_marker, history_days)

primary_frame = city_frames[primary_city].copy()
latest_row = _latest_record(primary_frame)
latest_category = aqi_to_category(latest_row["aqi"])
latest_color = get_category_color(latest_category)
latest_action = get_recommended_action(latest_category)
last_updated = pd.to_datetime(primary_frame["timestamp"].max(), utc=True).strftime("%Y-%m-%d %H:%M UTC")

st.sidebar.markdown(f"**Batch refresh:** data is refreshed periodically, not live.\n\n**Last updated:** {last_updated}")

st.subheader("Current AQI and advisories")
card_columns = st.columns(len(selected_cities))
for column, city in zip(card_columns, selected_cities):
    city_frame = city_frames[city]
    city_latest = _latest_record(city_frame)
    city_category = aqi_to_category(city_latest["aqi"])
    with column:
        _render_card(city, f"{int(round(city_latest['aqi']))}", f"{city_category} | {get_recommended_action(city_category)}", get_category_color(city_category))

st.markdown("### Historical trend")
st.plotly_chart(_trend_chart(city_frames, primary_city), width="stretch")

peak_stats = compute_peak_offpeak_stats(primary_frame)
partial_stats = partial_correlation_controlling_weather(primary_frame)
st.markdown("### Peak vs. off-peak correlation")
left, middle, right = st.columns(3)
with left:
    st.metric("Peak mean AQI", f"{peak_stats.get('peak', {}).get('mean', 0):.1f}" if peak_stats.get("peak") else "n/a")
    st.metric("Peak median AQI", f"{peak_stats.get('peak', {}).get('median', 0):.1f}" if peak_stats.get("peak") else "n/a")
with middle:
    st.metric("Off-peak mean AQI", f"{peak_stats.get('off_peak', {}).get('mean', 0):.1f}" if peak_stats.get("off_peak") else "n/a")
    st.metric("Test used", peak_stats.get("test_used", "n/a"))
with right:
    p_value = peak_stats.get("p_value")
    st.metric("p-value", f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "n/a")
    st.metric("Partial r", f"{partial_stats.get('correlation'):.3f}" if isinstance(partial_stats.get("correlation"), (int, float)) else "n/a")

st.write(_analysis_sentence(peak_stats, partial_stats))
st.caption(f"Weather-controlled interpretation: {partial_stats.get('interpretation', 'Unavailable')}")

model = load_city_model(primary_city, batch_marker, history_days)
forecast = predict_next_hours(model, hours=forecast_hours)
forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True)
st.markdown("### Forecast panel")
st.plotly_chart(_forecast_chart(forecast), width="stretch")
if hasattr(model, "evaluation_"):
    evaluation = getattr(model, "evaluation_", {})
    eval_left, eval_right = st.columns(2)
    with eval_left:
        st.metric("Holdout MAE", f"{evaluation.get('mae', 0):.1f}" if evaluation.get("mae") is not None else "n/a")
    with eval_right:
        st.metric("Holdout RMSE", f"{evaluation.get('rmse', 0):.1f}" if evaluation.get("rmse") is not None else "n/a")

st.markdown("### Seasonal heatmap")
st.plotly_chart(_heatmap_figure(primary_frame), width="stretch")

alerts_frame = flag_severe_hours(primary_frame)
alerts_summary = summarize_alerts(primary_frame)
st.markdown("### Alerts panel")
alert_columns = st.columns(3)
with alert_columns[0]:
    st.metric("Flagged hours", alerts_summary["flagged_hours"])
with alert_columns[1]:
    st.metric("Flagged share", f"{alerts_summary['flagged_share_pct']:.1f}%")
with alert_columns[2]:
    st.metric("Peak share of flagged", f"{alerts_summary['peak_flagged_share_pct']:.1f}%")

if alerts_frame.empty:
    st.info("No severe AQI hours were flagged in the selected window.")
else:
    display_alerts = alerts_frame[["timestamp", "aqi", "aqi_category", "is_peak_hour", "temperature_c", "humidity_pct", "wind_speed_mps"]].copy()
    display_alerts["timestamp"] = pd.to_datetime(display_alerts["timestamp"], utc=True)
    st.dataframe(display_alerts.sort_values("timestamp", ascending=False), width="stretch", height=280)

report_payload = {
    "interpretation": peak_stats.get("interpretation", "Unavailable"),
    "partial": partial_stats,
    "alerts": alerts_summary,
}
report_text = generate_summary_report(primary_city, report_payload, forecast, alerts_frame)
st.download_button(
    label="Download summary report",
    data=report_text,
    file_name=f"airsense_{primary_city.lower()}_summary.txt",
    mime="text/plain",
)

st.caption(
    f"Latest AQI in {primary_city}: {int(round(latest_row['aqi']))} ({latest_category}). {latest_action}"
)
