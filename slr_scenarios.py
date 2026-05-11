"""
NOAA 2022 Sea-Level Rise scenarios for Virginia Key Station 8723214
(Biscayne Bay, Miami, FL).  Interpolates anchor values to annual steps,
writes a CSV, and saves a plot.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 1. Anchor values — NOAA 2022 SLR Technical Report, Station 8723214
#    Units: feet above 2000 baseline
# ---------------------------------------------------------------------------
ANCHORS = {
    "slr_low_ft": {
        2020: 0.1, 2030: 0.2, 2040: 0.3, 2050: 0.4, 2060: 0.5, 2075: 0.7,
    },
    "slr_intermediate_ft": {
        2020: 0.2, 2030: 0.4, 2040: 0.6, 2050: 1.0, 2060: 1.4, 2075: 2.0,
    },
    "slr_high_ft": {
        2020: 0.3, 2030: 0.6, 2040: 1.0, 2050: 1.6, 2060: 2.2, 2075: 3.2,
    },
}

SCENARIO_STYLE = {
    "slr_low_ft":          {"label": "Low",          "color": "#2196F3", "lw": 2.0},
    "slr_intermediate_ft": {"label": "Intermediate",  "color": "#FF9800", "lw": 2.0},
    "slr_high_ft":         {"label": "High",          "color": "#F44336", "lw": 2.0},
}

# ---------------------------------------------------------------------------
# 2. Build a continuous 2025-2075 index and interpolate
# ---------------------------------------------------------------------------
years = np.arange(2025, 2076)          # 2025 … 2075 inclusive
df = pd.DataFrame({"year": years}).set_index("year")

for col, anchor_dict in ANCHORS.items():
    # Insert anchor points outside the target window so edge values interpolate
    # correctly (2020 anchor anchors the 2025 start; 2075 is the right bound).
    anchor_series = pd.Series(anchor_dict, name=col)
    anchor_series.index.name = "year"
    df[col] = anchor_series          # NaN where anchor not present
    # Re-index to union of continuous years + anchor years, interpolate, then slice
    full_index = sorted(set(years) | set(anchor_dict.keys()))
    s = anchor_series.reindex(full_index).interpolate(method="linear")
    df[col] = s.reindex(years).values

df = df.reset_index()                 # year back as a regular column
df = df[["year", "slr_low_ft", "slr_intermediate_ft", "slr_high_ft"]]

# ---------------------------------------------------------------------------
# 3. Save CSV
# ---------------------------------------------------------------------------
out_dir = pathlib.Path("outputs")
data_dir = pathlib.Path("data")
out_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

csv_path = data_dir / "slr_scenarios.csv"
df.to_csv(csv_path, index=False)
print(f"Saved CSV → {csv_path}\n")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for col, style in SCENARIO_STYLE.items():
    ax.plot(df["year"], df[col],
            label=style["label"], color=style["color"], linewidth=style["lw"])

# Reference lines
ax.axvline(2050, color="gray", linestyle="--", linewidth=0.9, zorder=1)
ax.axhline(0,    color="black", linestyle="--", linewidth=0.8, zorder=1)

# 2050 annotations
row_2050 = df.loc[df["year"] == 2050].iloc[0]
for col, style in SCENARIO_STYLE.items():
    val = row_2050[col]
    ax.annotate(
        f"{val:.1f} ft",
        xy=(2050, val),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=style["color"],
    )

ax.set_xlim(2025, 2075)
ax.set_ylim(bottom=-0.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Sea-Level Rise (feet above 2000 baseline)", fontsize=11)
ax.set_title(
    "NOAA 2022 Sea-Level Rise Scenarios — Virginia Key, Miami FL",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
png_path = out_dir / "slr_scenarios.png"
fig.savefig(png_path, dpi=200)
print(f"Saved plot → {png_path}\n")

# ---------------------------------------------------------------------------
# 5. Print head and tail
# ---------------------------------------------------------------------------
print("--- DataFrame head ---")
print(df.head(6).to_string(index=False))
print("\n--- DataFrame tail ---")
print(df.tail(6).to_string(index=False))
