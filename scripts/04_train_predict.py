import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

os.makedirs("data/models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ---------- INPUT FILES ----------
FEATURES_PATH = "data/processed/features_history.parquet"
LOAD_PATH = "data/processed/load_hourly.parquet"
WEATHER_FORECAST_PATH = "data/processed/weather_forecast.parquet"

# ---------- OUTPUT FILES ----------
MODEL_PATH = "data/models/rf_model.pkl"
FI_OUT = "data/processed/feature_importance.parquet"
EVAL_OUT = "data/processed/predictions_eval.parquet"
TOMORROW_OUT = "data/processed/tomorrow_predictions.parquet"

# ---------- TRAINING FEATURES ----------
TARGET = "load"
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "hour",
    "dayofweek",
    "is_weekend",
    "load_lag_1",
    "load_lag_24",
    "load_roll_24",
]

# ---------- LOAD TRAINING DATA ----------
df = pd.read_parquet(FEATURES_PATH).copy()
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

missing_cols = [c for c in ["time", TARGET] + FEATURES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in {FEATURES_PATH}: {missing_cols}")

X = df[FEATURES]
y = df[TARGET]

# ---------- TRAIN MODEL ----------
model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print(f"✅ Saved model -> {MODEL_PATH}")

# ---------- FEATURE IMPORTANCE (drivers chart) ----------
fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
fi.to_parquet(FI_OUT, index=False)
print(f"✅ Saved feature importance -> {FI_OUT}")

# ---------- EVALUATION PREDICTIONS FOR ALL HISTORY (so accuracy works for any date) ----------
df_eval = df[["time", "load"]].copy()
df_eval["predicted_load"] = model.predict(df[FEATURES])
df_eval.to_parquet(EVAL_OUT, index=False)
print(f"✅ Saved evaluation predictions -> {EVAL_OUT}")

# ---------- LOAD HISTORY + WEATHER FORECAST ----------
load_hist = pd.read_parquet(LOAD_PATH).copy()
load_hist["time"] = pd.to_datetime(load_hist["time"])
load_hist = load_hist.sort_values("time").reset_index(drop=True)

weather_fc = pd.read_parquet(WEATHER_FORECAST_PATH).copy()
weather_fc["time"] = pd.to_datetime(weather_fc["time"])
weather_fc = weather_fc.sort_values("time").reset_index(drop=True)

# ---------- KEEP ONLY TOMORROW ----------
tomorrow_date = (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).date()
weather_tomorrow = weather_fc[weather_fc["time"].dt.date == tomorrow_date].copy()

if weather_tomorrow.empty:
    raise ValueError(
        "No rows found for tomorrow in weather_forecast.parquet.\n"
        "Fix: re-run python scripts/02b_fetch_weather_forecast.py and try again."
    )

# ---------- ROLLING TOMORROW FORECAST (fixes straight line) ----------
series = load_hist["load"].astype(float).tolist()

future_rows = []
for i in range(len(weather_tomorrow)):
    t = weather_tomorrow.iloc[i]["time"]

    lag_1 = series[-1]
    lag_24 = series[-24] if len(series) >= 24 else series[-1]
    roll_24 = sum(series[-24:]) / min(24, len(series))

    row = {
        "temperature_2m": float(weather_tomorrow.iloc[i]["temperature_2m"]),
        "relative_humidity_2m": float(weather_tomorrow.iloc[i]["relative_humidity_2m"]),
        "precipitation": float(weather_tomorrow.iloc[i]["precipitation"]),
        "hour": t.hour,
        "dayofweek": t.dayofweek,
        "is_weekend": int(t.dayofweek >= 5),
        "load_lag_1": float(lag_1),
        "load_lag_24": float(lag_24),
        "load_roll_24": float(roll_24),
    }

    pred = float(model.predict(pd.DataFrame([row])[FEATURES])[0])

    # KEY STEP: use this prediction as input for next hour
    series.append(pred)

    future_rows.append({"time": t, "predicted_load": pred})

tomorrow_df = pd.DataFrame(future_rows)
tomorrow_df.to_parquet(TOMORROW_OUT, index=False)

print("✅ Tomorrow prediction generated (rolling forecast)")
print(f"Saved -> {TOMORROW_OUT}")
print("Range:", tomorrow_df["time"].min(), "→", tomorrow_df["time"].max())
print(tomorrow_df.head())
print(tomorrow_df.tail())
