from gridstatus import NYISO
import pandas as pd
from datetime import datetime, timedelta
import os

os.makedirs("data/processed", exist_ok=True)

OUT_PATH = "data/processed/load_hourly.parquet"
LOOKBACK_DAYS = 14

iso = NYISO()
end = datetime.utcnow()
start = end - timedelta(days=LOOKBACK_DAYS)

print(f"Fetching NYISO load from {start} to {end} ...")
df = iso.get_load(start=start, end=end)

# ---- Make sure time is a column ----
# Sometimes gridstatus returns time as index or with different column names.
df = df.reset_index()  # safe even if index isn't time
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

print("Columns returned:", df.columns.tolist())

# Find the time column
possible_time_cols = ["time", "timestamp", "interval_start", "datetime", "date"]
time_col = next((c for c in possible_time_cols if c in df.columns), None)

if time_col is None:
    # last resort: pick first datetime-like column
    for c in df.columns:
        if "time" in c or "interval" in c or "date" in c:
            time_col = c
            break

if time_col is None:
    raise ValueError(f"Couldn't find a time column. Columns were: {df.columns.tolist()}")

# Standardize
df = df.rename(columns={time_col: "time"})
df["time"] = pd.to_datetime(df["time"])

# Find the load column (sometimes it's 'load', sometimes 'mw', etc.)
possible_load_cols = ["load", "mw", "value"]
load_col = next((c for c in possible_load_cols if c in df.columns), None)

if load_col is None:
    # try to find something that looks like load
    for c in df.columns:
        if "load" in c or "mw" in c:
            load_col = c
            break

if load_col is None:
    raise ValueError(f"Couldn't find load column. Columns were: {df.columns.tolist()}")

df = df.rename(columns={load_col: "load"})

# Resample to hourly
hourly = (
    df.set_index("time")[["load"]]
    .resample("h")
    .mean()
    .reset_index()
)

hourly = hourly.drop_duplicates(subset=["time"]).sort_values("time")

# Append + dedupe (keeps history growing)
if os.path.exists(OUT_PATH):
    old = pd.read_parquet(OUT_PATH)
    old["time"] = pd.to_datetime(old["time"])

    combined = pd.concat([old, hourly], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
else:
    combined = hourly.reset_index(drop=True)

combined.to_parquet(OUT_PATH, index=False)

print(f"✅ Saved hourly load history -> {OUT_PATH}")
print("Rows:", len(combined))
print("Range:", combined["time"].min(), "to", combined["time"].max())
print(combined.tail(5))
