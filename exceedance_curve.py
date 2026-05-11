"""
Tidal exceedance frequency curve for Virginia Key Station 8723214, 2023.
Loads cleaned hourly water levels, builds an exceedance table, plots the
curve, and reports flood hours for a 1.5 ft NAVD88 parcel under current and
2050 SLR scenarios.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Reference elevations / SLR offsets (ft)
# ---------------------------------------------------------------------------
MHHW_FT       = 0.87
PARCEL_FT     = 1.50
SLR_2050_INT  = 1.00   # Intermediate 2050 (from slr_scenarios.py)
SLR_2050_HIGH = 1.60   # High 2050

VLINES = [
    (MHHW_FT, "MHHW\n0.87 ft",  "#F44336"),
    (1.50,    "1.50 ft",         "#FF9800"),
    (2.00,    "2.00 ft",         "#9C27B0"),
]

# ---------------------------------------------------------------------------
# 1. Load water-level data
# ---------------------------------------------------------------------------
csv_path = pathlib.Path("data/noaa_wl_2023.csv")
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found — run noaa_tides.py first.")
    sys.exit(1)

df = pd.read_csv(csv_path, parse_dates=["datetime"])

if df["wl_ft_navd88"].isna().any():
    n_na = df["wl_ft_navd88"].isna().sum()
    print(f"Warning: {n_na} NaN values remain after loading; dropping them.")
    df = df.dropna(subset=["wl_ft_navd88"])

n_obs = len(df)
print(f"Loaded {n_obs:,} hourly observations from {csv_path}.")

# ---------------------------------------------------------------------------
# 2. Build exceedance table
# ---------------------------------------------------------------------------
thresholds = np.arange(-1.00, 4.001, 0.05)   # -1.0 to 4.0 in 0.05 ft steps

wl = df["wl_ft_navd88"].values

records = [
    {"threshold_ft": round(t, 2), "hours_per_year": int((wl > t).sum())}
    for t in thresholds
]
exc = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
out_dir = pathlib.Path("outputs")
out_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(exc["threshold_ft"], exc["hours_per_year"],
        color="#1565C0", linewidth=2.0, zorder=3)

for x_val, label, color in VLINES:
    ax.axvline(x_val, color=color, linestyle="--", linewidth=1.1, zorder=2)
    # Look up the hours value at the nearest threshold for label placement
    idx  = (exc["threshold_ft"] - x_val).abs().idxmin()
    y_at = exc.loc[idx, "hours_per_year"]
    # Place label just above the curve intersection
    ax.text(x_val + 0.04, max(y_at * 1.3, 2), label,
            color=color, fontsize=8.5, va="bottom")

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

ax.set_xlim(thresholds[0], thresholds[-1])
ax.set_ylim(bottom=0.5)
ax.set_xlabel("Water Level Threshold (ft, NAVD88)", fontsize=11)
ax.set_ylabel("Hours per Year Exceeding Threshold  [log scale]", fontsize=11)
ax.set_title(
    "Tidal Exceedance Frequency — Virginia Key 2023 (NAVD88)",
    fontsize=12, fontweight="bold",
)
ax.grid(which="both", axis="y", alpha=0.25)
ax.grid(axis="x", alpha=0.15)

fig.tight_layout()
png_path = out_dir / "exceedance_curve.png"
fig.savefig(png_path, dpi=200)
print(f"Saved plot → {png_path}\n")

# ---------------------------------------------------------------------------
# 4. Flood-hours report for a 1.5 ft NAVD88 parcel
# ---------------------------------------------------------------------------
def hours_exceeding(water_levels, threshold):
    """Count hours water level exceeds threshold."""
    return int((water_levels > threshold).sum())

def lookup_exc(threshold):
    """Interpolate hours/year from the exceedance table for any threshold."""
    idx = (exc["threshold_ft"] - threshold).abs().idxmin()
    return exc.loc[idx, "hours_per_year"]

# Under SLR the sea surface rises, so the *effective* threshold drops by the
# SLR amount (equivalent to the parcel being that much lower relative to the sea).
h_current  = hours_exceeding(wl, PARCEL_FT)
h_int_2050 = hours_exceeding(wl, PARCEL_FT - SLR_2050_INT)
h_hi_2050  = hours_exceeding(wl, PARCEL_FT - SLR_2050_HIGH)

print("=" * 55)
print(f"  Parcel elevation: {PARCEL_FT} ft NAVD88")
print("=" * 55)
print(f"  2023 baseline (no SLR)      : {h_current:>5,} hrs/yr")
print(f"  2050 Intermediate (+{SLR_2050_INT:.1f} ft SLR): {h_int_2050:>5,} hrs/yr")
print(f"  2050 High         (+{SLR_2050_HIGH:.1f} ft SLR): {h_hi_2050:>5,} hrs/yr")
print("=" * 55)

# ---------------------------------------------------------------------------
# 5. Save exceedance CSV
# ---------------------------------------------------------------------------
data_dir = pathlib.Path("data")
data_dir.mkdir(exist_ok=True)
exc_csv = data_dir / "exceedance_curve.csv"
exc.to_csv(exc_csv, index=False)
print(f"\nSaved exceedance table → {exc_csv}")
print(f"  {len(exc):,} rows  |  threshold range: "
      f"{exc['threshold_ft'].min()} to {exc['threshold_ft'].max()} ft")
