import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

# ---------- INPUT FILES ----------
LOAD_PATH = "data/processed/load_hourly.parquet"
WEATHER_PATH = "data/processed/weather_hourly.parquet"

# ---------- OUTPUT FILE ----------
OUT_PATH = "data/processed/features_history.parquet"

# ---------- LOAD DATA ----------
load_df = pd.read_parquet(LOAD_PATH)
weather_df = pd.read_parquet(WEATHER_PATH)

# Ensure datetime
load_df["time"] = pd.to_datetime(load_df["time"])
weather_df["time"] = pd.to_datetime(weather_df["time"])

# Make weather timezone-naive to match load (safe choice)
weather_df["time"] = weather_df["time"].dt.tz_localize(None)
load_df["time"] = load_df["time"].dt.tz_localize(None)

# ---------- JOIN ----------
df = pd.merge(load_df, weather_df, on="time", how="inner")

# ---------- FEATURE ENGINEERING ----------
df = df.sort_values("time").reset_index(drop=True)

df["hour"] = df["time"].dt.hour
df["dayofweek"] = df["time"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

df["load_lag_1"] = df["load"].shift(1)
df["load_lag_24"] = df["load"].shift(24)
df["load_roll_24"] = df["load"].rolling(24).mean()

# Drop rows that can't have lags yet
df = df.dropna().reset_index(drop=True)

# ---------- SAVE ----------
df.to_parquet(OUT_PATH, index=False)

print("✅ Features built successfully")
print("Saved to:", OUT_PATH)
print("Rows:", len(df))
print("Range:", df["time"].min(), "→", df["time"].max())
print(df.head())
