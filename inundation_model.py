"""
Inundation model for Brickell / Biscayne Bay study area.

Physics:
  Stillwater depth   : d(x,t) = BFE + SLR(t) - elev(x)
  Chronic inundation : elev(x) < MHHW + SLR(t)
  Onset year T_c(x)  : first t where elev(x) - MHHW < SLR(t)
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PARCELS_IN  = pathlib.Path("data/study_parcels_full.gpkg")
SLR_CSV     = pathlib.Path("data/slr_scenarios.csv")
PARCELS_OUT = pathlib.Path("data/study_parcels_physics.gpkg")
PLOT_OUT    = pathlib.Path("outputs/Tc_histograms.png")

MHHW_FT   = 0.87     # NAVD88, Virginia Key station 8723214
SAFE_YEAR = 2076     # sentinel: chronic inundation does not occur by 2075

SLR_COLS = {
    "low":          "slr_low_ft",
    "intermediate": "slr_intermediate_ft",
    "high":         "slr_high_ft",
}
TC_COLS = {k: f"T_c_{k}" for k in SLR_COLS}

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
for path in (PARCELS_IN, SLR_CSV):
    if not path.exists():
        script = "spatial_join_fema.py" if "parcels" in path.name else "slr_scenarios.py"
        print(f"ERROR: {path} not found — run {script} first.")
        sys.exit(1)

print("Loading data …")
gdf = gpd.read_file(str(PARCELS_IN))
slr = pd.read_csv(str(SLR_CSV))
slr = slr.set_index("year")          # index = year (int), columns = slr_*_ft

print(f"  {len(gdf):,} parcels loaded")
print(f"  SLR scenarios: years {slr.index.min()}–{slr.index.max()}\n")

# ---------------------------------------------------------------------------
# 2. compute_Tc
# ---------------------------------------------------------------------------
def compute_Tc(elev_ft: float, slr_series: pd.Series, mhhw_ft: float = MHHW_FT) -> int:
    """
    Return the first year where elev_ft - mhhw_ft < slr_series[year].
    Returns SAFE_YEAR (2076) if chronic inundation never occurs in 2025-2075.
    """
    headroom = elev_ft - mhhw_ft
    # slr_series is indexed by year; find first year SLR exceeds headroom
    exceeded = slr_series.index[slr_series > headroom]
    return int(exceeded[0]) if len(exceeded) > 0 else SAFE_YEAR


# ---------------------------------------------------------------------------
# 3. Apply compute_Tc for all three scenarios
# ---------------------------------------------------------------------------
print("Computing T_c for all parcels and scenarios …")

for scenario_key, slr_col in SLR_COLS.items():
    slr_series = slr[slr_col]                        # Series indexed by year
    tc_col = TC_COLS[scenario_key]
    gdf[tc_col] = gdf["elev_ft_navd88"].apply(
        lambda e: compute_Tc(e, slr_series) if pd.notna(e) else SAFE_YEAR
    )
    print(f"  {tc_col}: done")

print()

# ---------------------------------------------------------------------------
# 4. compute_d100yr
# ---------------------------------------------------------------------------
def compute_d100yr(bfe_ft: float, slr_ft: float, elev_ft: float) -> float:
    """100-year stillwater inundation depth (ft). Positive = inundated."""
    return bfe_ft + slr_ft - elev_ft


# ---------------------------------------------------------------------------
# 5. 100-year flood depths at four time points (Intermediate scenario)
# ---------------------------------------------------------------------------
print("Computing 100-year inundation depths (Intermediate SLR) …")

TIME_POINTS = [2025, 2040, 2050, 2075]
col_names   = ["d_100yr_2025", "d_100yr_2040_int", "d_100yr_2050_int", "d_100yr_2075_int"]

for year, col in zip(TIME_POINTS, col_names):
    slr_val = slr.loc[year, SLR_COLS["intermediate"]]
    gdf[col] = gdf.apply(
        lambda row: compute_d100yr(row["bfe_ft"], slr_val, row["elev_ft_navd88"])
        if pd.notna(row["bfe_ft"]) and pd.notna(row["elev_ft_navd88"])
        else np.nan,
        axis=1,
    )
    n_inundated = (gdf[col] > 0).sum()
    print(f"  {year} (+{slr_val:.2f} ft SLR): {n_inundated:,} parcels inundated")

print()

# ---------------------------------------------------------------------------
# 6. already_flooded_2025
# ---------------------------------------------------------------------------
gdf["already_flooded_2025"] = gdf["d_100yr_2025"] > 0

# ---------------------------------------------------------------------------
# 7. chronic_by_2050_intermediate
# ---------------------------------------------------------------------------
gdf["chronic_by_2050_intermediate"] = gdf[TC_COLS["intermediate"]] <= 2050

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
n_total    = len(gdf)
n_flooded  = gdf["already_flooded_2025"].sum()

print("=" * 58)
print(f"  Total parcels analyzed          : {n_total:>6,}")
print(f"  Already below BFE in 2025       : {n_flooded:>6,}"
      f"  ({100*n_flooded/n_total:.1f}%)")
print()

for scenario_key in SLR_COLS:
    tc_col   = TC_COLS[scenario_key]
    n_before = (gdf[tc_col] <= 2050).sum()
    # Median only for parcels that do reach chronic inundation
    chronic  = gdf.loc[gdf[tc_col] < SAFE_YEAR, tc_col]
    med      = int(chronic.median()) if len(chronic) > 0 else "N/A"
    print(f"  {scenario_key.capitalize():<14} scenario:")
    print(f"    Chronically flooded by 2050 : {n_before:>5,}")
    print(f"    Median T_c (excl. safe)     : {med}")

print("=" * 58)
print()

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
pathlib.Path("data").mkdir(exist_ok=True)
gdf.to_file(str(PARCELS_OUT), driver="GPKG")
print(f"Saved → {PARCELS_OUT}  ({len(gdf):,} features)")

# ---------------------------------------------------------------------------
# 10. T_c histograms — three side-by-side subplots
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)

SCENARIO_LABELS = {
    "low":          "Low SLR",
    "intermediate": "Intermediate SLR",
    "high":         "High SLR",
}
COLORS = {"low": "#2196F3", "intermediate": "#FF9800", "high": "#F44336"}
BINS   = np.arange(2025, 2086, 10)   # 2025, 2035, 2045, 2055, 2065, 2075, 2085

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, scenario_key in zip(axes, SLR_COLS):
    tc_col = TC_COLS[scenario_key]
    # Include SAFE_YEAR parcels in the rightmost bin to show "never flooded"
    data = gdf[tc_col].values

    ax.hist(data, bins=BINS, color=COLORS[scenario_key],
            edgecolor="white", linewidth=0.5)
    ax.axvline(2050, color="black", linestyle="--", linewidth=1.2)
    ax.text(2051, ax.get_ylim()[1] * 0.95, "2050",
            fontsize=8, va="top", color="black")

    n_safe    = (data == SAFE_YEAR).sum()
    n_chronic = (data < SAFE_YEAR).sum()
    ax.set_title(
        f"{SCENARIO_LABELS[scenario_key]}\n"
        f"({n_chronic} chronic, {n_safe} safe by 2075)",
        fontsize=10,
    )
    ax.set_xlabel("Year of Chronic Inundation Onset", fontsize=9)
    ax.set_xlim(2020, 2085)
    ax.set_xticks(BINS[:-1])
    ax.tick_params(axis="x", labelsize=8)

axes[0].set_ylabel("Number of Parcels", fontsize=10)

fig.suptitle(
    "Years to Chronic Inundation — Brickell Study Area",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig(str(PLOT_OUT), dpi=200, bbox_inches="tight")
print(f"Saved plot → {PLOT_OUT}")
