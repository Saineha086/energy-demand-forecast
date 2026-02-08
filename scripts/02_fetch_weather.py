import os, json
import pandas as pd
import requests
from datetime import datetime, timedelta

os.makedirs("data/processed", exist_ok=True)

OUT_PATH = "data/processed/weather_hourly.parquet"

# Read config (create config.json if you haven't)
# Example config.json:
# {
#   "latitude": 40.7128,
#   "longitude": -74.0060,
#   "timezone": "America/New_York",
#   "lookback_days": 14
# }
cfg = json.load(open("config.json"))
LAT = cfg["latitude"]
LON = cfg["longitude"]
TZ = cfg["timezone"]
LOOKBACK_DAYS = int(cfg.get("lookback_days", 14))

end_date = datetime.now().date()
start_date = end_date - timedelta(days=LOOKBACK_DAYS)

url = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LAT}&longitude={LON}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation"
    f"&timezone={TZ}"
)

print("Fetching weather:", url)
r = requests.get(url, timeout=60)
r.raise_for_status()
data = r.json()

df = pd.DataFrame(data["hourly"])
df.columns = [c.strip().lower() for c in df.columns]
df["time"] = pd.to_datetime(df["time"])

# Open-Meteo gives local timezone times but pandas reads them as naive.
# We'll localize later when merging. For storage, keep it consistent.
df = df.drop_duplicates(subset=["time"]).sort_values("time")

# Append + dedupe (keeps history growing)
if os.path.exists(OUT_PATH):
    old = pd.read_parquet(OUT_PATH)
    old["time"] = pd.to_datetime(old["time"])
    combined = pd.concat([old, df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
else:
    combined = df.reset_index(drop=True)

combined.to_parquet(OUT_PATH, index=False)

print(f"✅ Saved weather history -> {OUT_PATH}")
print("Rows:", len(combined))
print("Range:", combined['time'].min(), "to", combined['time'].max())
print(combined.tail(5))
