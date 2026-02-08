import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import os

st.set_page_config(page_title="Energy Demand Dashboard", layout="wide")
st.title("⚡ Energy Demand Forecast & Insights (NYISO)")
st.caption("Historical actual demand + model accuracy view + tomorrow forecast")

# ----------------------------
# Helper: normalize date input
# ----------------------------
def normalize_date_range(picked):
    """
    Streamlit date_input can return:
    - a single date
    - a tuple/list of 2 dates
    - a tuple with (start, None) while user is still selecting
    This function always returns (start_date, end_date) as date objects.
    """
    if isinstance(picked, (list, tuple)):
        # if user is mid-selection, Streamlit can give (start, None)
        if len(picked) == 2:
            start, end = picked[0], picked[1]
            if end is None:
                end = start
            return start, end
        # sometimes it can be length 1
        if len(picked) == 1:
            return picked[0], picked[0]
        # fallback
        return picked[0], picked[0]
    else:
        return picked, picked

# ----------------------------
# Load data
# ----------------------------
actuals = pd.read_parquet("data/processed/load_hourly.parquet")
actuals["time"] = pd.to_datetime(actuals["time"])
actuals = actuals.sort_values("time")

# Try to load eval predictions (for two-line chart)
pred_eval = None
pred_paths = [
    "data/processed/predictions.parquet",        # older file you had
    "data/processed/predictions_eval.parquet",   # if you create later
]
pred_path_found = next((p for p in pred_paths if os.path.exists(p)), None)
if pred_path_found:
    pred_eval = pd.read_parquet(pred_path_found)
    pred_eval["time"] = pd.to_datetime(pred_eval["time"])
    pred_eval = pred_eval.sort_values("time")

# Tomorrow forecast
forecast = pd.read_parquet("data/processed/tomorrow_predictions.parquet")
forecast["time"] = pd.to_datetime(forecast["time"])
forecast = forecast.sort_values("time")

# Feature importance (drivers)
fi = None
fi_path = "data/processed/feature_importance.parquet"
if os.path.exists(fi_path):
    fi = pd.read_parquet(fi_path)

# ----------------------------
# Sidebar filters (FIXED)
# ----------------------------
st.sidebar.header("Filters")

min_day = actuals["time"].min().date()
max_day = actuals["time"].max().date()

picked = st.sidebar.date_input(
    "Select historical date range",
    value=(max_day - timedelta(days=7), max_day),
    min_value=min_day,
    max_value=max_day
)

start_day, end_day = normalize_date_range(picked)

# Safety: if user picked reversed order
if start_day > end_day:
    start_day, end_day = end_day, start_day

# Filter actuals
hist = actuals[
    (actuals["time"].dt.date >= start_day) &
    (actuals["time"].dt.date <= end_day)
].copy()

# ----------------------------
# SECTION 1: Historical actual demand
# ----------------------------
st.subheader("📈 Historical Actual Demand")
if hist.empty:
    st.warning("No actual data found for this date range. Try a different range.")
else:
    st.line_chart(hist.set_index("time")[["load"]], use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Load", f"{hist['load'].mean():.0f} MW")
    c2.metric("Max Load", f"{hist['load'].max():.0f} MW")
    c3.metric("Min Load", f"{hist['load'].min():.0f} MW")

# ----------------------------
# SECTION 2: Actual vs Predicted (Two lines)
# ----------------------------
st.subheader("✅ Actual vs Predicted (Model Accuracy View)")

if pred_eval is None:
    st.info(
        "I can't show the 2-line chart because I can't find predictions.parquet.\n\n"
        "Fix: make sure one of these exists:\n"
        "- data/processed/predictions.parquet\n"
        "- data/processed/predictions_eval.parquet"
    )
else:
    pe = pred_eval[
        (pred_eval["time"].dt.date >= start_day) &
        (pred_eval["time"].dt.date <= end_day)
    ].copy()

    if pe.empty:
        st.warning("No prediction data available for the selected date range. Try a more recent range.")
    else:
        # Ensure expected columns exist
        needed = {"load", "predicted_load"}
        if not needed.issubset(set(pe.columns)):
            st.error(f"Predictions file is missing columns: {needed - set(pe.columns)}")
        else:
            st.line_chart(pe.set_index("time")[["load", "predicted_load"]], use_container_width=True)

            pe["abs_error"] = (pe["load"] - pe["predicted_load"]).abs()
            st.metric("Avg Absolute Error (selected range)", f"{pe['abs_error'].mean():.1f} MW")

# ----------------------------
# SECTION 3: Demand drivers
# ----------------------------
st.subheader("🧠 What Drives Electricity Demand?")
if fi is None:
    st.info("feature_importance.parquet not found. Run: python scripts/04_train_predict.py")
else:
    top = fi.sort_values("importance", ascending=False).head(10).copy()
    st.bar_chart(top.set_index("feature")["importance"], use_container_width=True)

# ----------------------------
# SECTION 4: Tomorrow forecast
# ----------------------------
st.subheader("🔮 Tomorrow Forecast (Next-Day Prediction)")
tomorrow_date = forecast["time"].dt.date.min()
st.write(f"Forecast date: **{tomorrow_date}**")

st.line_chart(forecast.set_index("time")[["predicted_load"]], use_container_width=True)

peak_t = forecast.loc[forecast["predicted_load"].idxmax()]
st.warning(
    f"⚠️ Predicted peak tomorrow around **{peak_t['time'].strftime('%Y-%m-%d %H:%M')}** "
    f"with load ≈ **{peak_t['predicted_load']:.0f} MW**"
)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
