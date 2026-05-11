"""
Pull hourly verified water-level data from NOAA CO-OPS for Virginia Key
Station 8723214 (Biscayne Bay, Miami FL) for the full year 2023.
Cleans, interpolates, saves, and plots the time series.
"""

import pathlib
import sys

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_URL  = "https://tidesandcurrents.noaa.gov/api/datagetter"
STATION  = "8723214"
MHHW_FT  = 0.87          # MHHW in ft NAVD88 for Virginia Key
TOTAL_HRS = 8760          # hours in 2023 (non-leap year)

PARAMS = {
    "begin_date": "20230101",
    "end_date":   "20231231",
    "station":    STATION,
    "product":    "water_level",
    "datum":      "NAVD",
    "units":      "english",
    "time_zone":  "GMT",
    "format":     "json",
}

# ---------------------------------------------------------------------------
# 1. Fetch data
# ---------------------------------------------------------------------------
print(f"Fetching hourly water-level data from NOAA CO-OPS (station {STATION})…")
try:
    response = requests.get(API_URL, params=PARAMS, timeout=60)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
    sys.exit(1)
except requests.exceptions.ConnectionError:
    print("Connection error — check your network and try again.")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("Request timed out — the NOAA API did not respond within 60 s.")
    sys.exit(1)
except requests.exceptions.RequestException as e:
    print(f"Unexpected request error: {e}")
    sys.exit(1)

payload = response.json()

# The CO-OPS API returns an 'error' key instead of a non-200 status on data
# errors (bad station, datum, date range, etc.).
if "error" in payload:
    print(f"NOAA API error: {payload['error'].get('message', payload['error'])}")
    sys.exit(1)

raw = payload.get("data", [])
if not raw:
    print("NOAA returned an empty data array — nothing to process.")
    sys.exit(1)

print(f"  Received {len(raw):,} raw records from the API.")

# ---------------------------------------------------------------------------
# 2 & 3. Parse → DataFrame, drop invalid values
# ---------------------------------------------------------------------------
df_raw = pd.DataFrame(raw)[["t", "v"]]
df_raw.columns = ["datetime", "wl_ft_navd88"]

df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], format="%Y-%m-%d %H:%M")

# Drop rows where the value field is empty or the literal string 'null'
invalid_mask = df_raw["wl_ft_navd88"].isin(["", "null"]) | df_raw["wl_ft_navd88"].isna()
df_clean = df_raw[~invalid_mask].copy()
df_clean["wl_ft_navd88"] = df_clean["wl_ft_navd88"].astype(float)

# ---------------------------------------------------------------------------
# 4. Report valid observations and missing hours
# ---------------------------------------------------------------------------
n_valid   = len(df_clean)
n_missing = TOTAL_HRS - n_valid

print(f"\n  Valid hourly observations : {n_valid:,}")
print(f"  Missing hours (of {TOTAL_HRS:,})  : {n_missing:,}")

# ---------------------------------------------------------------------------
# 5. Reindex to a complete hourly grid and interpolate gaps
# ---------------------------------------------------------------------------
full_index = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", name="datetime")
df_full = df_clean.set_index("datetime").reindex(full_index)

n_to_fill = df_full["wl_ft_navd88"].isna().sum()
df_full["wl_ft_navd88"] = df_full["wl_ft_navd88"].interpolate(method="linear")
print(f"  Gaps filled by interpolation: {n_to_fill:,}")

df_full = df_full.reset_index()
df_full.columns = ["datetime", "wl_ft_navd88"]

# ---------------------------------------------------------------------------
# 6. Save CSV
# ---------------------------------------------------------------------------
data_dir = pathlib.Path("data")
data_dir.mkdir(exist_ok=True)
csv_path = data_dir / "noaa_wl_2023.csv"
df_full.to_csv(csv_path, index=False)
print(f"\n  Saved CSV → {csv_path}")

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
out_dir = pathlib.Path("outputs")
out_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(df_full["datetime"], df_full["wl_ft_navd88"],
        linewidth=0.4, color="#1565C0", alpha=0.85, label="Water level")

ax.axhline(MHHW_FT, color="red", linestyle="--", linewidth=1.0,
           label=f"MHHW = {MHHW_FT} ft NAVD88")

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_xlim(df_full["datetime"].iloc[0], df_full["datetime"].iloc[-1])

ax.set_xlabel("2023", fontsize=11)
ax.set_ylabel("Water Level (ft, NAVD88)", fontsize=11)
ax.set_title(
    "NOAA CO-OPS Hourly Water Level — Virginia Key Station 8723214, 2023",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
png_path = out_dir / "noaa_wl_2023.png"
fig.savefig(png_path, dpi=200)
print(f"  Saved plot → {png_path}")
