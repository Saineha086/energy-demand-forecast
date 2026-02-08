import os, json
import pandas as pd
import requests
from datetime import datetime, timedelta

# Ensure output folder exists
os.makedirs("data/processed", exist_ok=True)

OUT_PATH = "data/processed/weather_forecast.parquet"

# Load config
cfg = json.load(open("config.json"))
LAT = cfg["latitude"]
LON = cfg["longitude"]
TZ = cfg["timezone"]

# We want tomorrow + next 24 hours
url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation"
    f"&timezone={TZ}"
    "&forecast_days=2"
)

print("Fetching tomorrow weather forecast...")
r = requests.get(url, timeout=60)
r.raise_for_status()
data = r.json()

df = pd.DataFrame(data["hourly"])
df.columns = [c.strip().lower() for c in df.columns]
df["time"] = pd.to_datetime(df["time"])

# Keep only future times (tomorrow onwards)
now = pd.Timestamp.now().tz_localize(None)
df = df[df["time"] > now]

# Save forecast
df.to_parquet(OUT_PATH, index=False)

print(f"✅ Saved tomorrow weather forecast -> {OUT_PATH}")
print("Rows:", len(df))
print(df.head())
print(df.tail())
