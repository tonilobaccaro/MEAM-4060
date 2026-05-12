"""
Inundation depth profiles for three Brickell parcel archetypes × three SLR
scenarios. Produces a 3×3 subplot grid with flooded/dry shading, T_c markers,
and a summary table.

Physics:  d_100yr(t) = BFE + SLR(t) - elev
          T_c        = first year where elev - MHHW < SLR(t)
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLR_CSV   = pathlib.Path("data/slr_scenarios.csv")
PLOT_OUT  = pathlib.Path("outputs/inundation_profiles.png")

BFE_FT    = 6.0    # representative AE-zone BFE for Brickell (ft NAVD88)
MHHW_FT   = 0.87   # Virginia Key, ft NAVD88
SAFE_YEAR = 2076   # sentinel — chronic inundation does not occur by 2075

ARCHETYPES = {
    "A: Low-lying\n(1.0 ft NAVD88)":  1.0,
    "B: Mid-range\n(2.5 ft NAVD88)":  2.5,
    "C: Elevated\n(4.0 ft NAVD88)":   4.0,
}

SCENARIOS = {
    "Low":          "slr_low_ft",
    "Intermediate": "slr_intermediate_ft",
    "High":         "slr_high_ft",
}

SCENARIO_COLORS = {
    "Low":          "#2196F3",
    "Intermediate": "#FF9800",
    "High":         "#F44336",
}

# ---------------------------------------------------------------------------
# Load SLR data
# ---------------------------------------------------------------------------
if not SLR_CSV.exists():
    print(f"ERROR: {SLR_CSV} not found — run slr_scenarios.py first.")
    sys.exit(1)

slr = pd.read_csv(str(SLR_CSV)).set_index("year")
years = slr.index.values    # 2025 … 2075

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def compute_Tc(elev_ft: float, slr_series: pd.Series) -> int:
    headroom = elev_ft - MHHW_FT
    exceeded = slr_series.index[slr_series > headroom]
    return int(exceeded[0]) if len(exceeded) > 0 else SAFE_YEAR


def compute_d100yr(bfe_ft: float, slr_array: np.ndarray, elev_ft: float) -> np.ndarray:
    return bfe_ft + slr_array - elev_ft


# ---------------------------------------------------------------------------
# Pre-compute all depths and T_c values
# ---------------------------------------------------------------------------
arch_labels   = list(ARCHETYPES.keys())
arch_elevs    = list(ARCHETYPES.values())
scenario_keys = list(SCENARIOS.keys())

# depths[arch_label][scenario] → np.ndarray shape (n_years,)
depths = {
    label: {
        sc: compute_d100yr(BFE_FT, slr[col].values, elev)
        for sc, col in SCENARIOS.items()
    }
    for label, elev in ARCHETYPES.items()
}

# tc[arch_label][scenario] → int year or SAFE_YEAR
tc = {
    label: {
        sc: compute_Tc(elev, slr[col])
        for sc, col in SCENARIOS.items()
    }
    for label, elev in ARCHETYPES.items()
}

# ---------------------------------------------------------------------------
# Print T_c summary table
# ---------------------------------------------------------------------------
def tc_str(val: int) -> str:
    return str(val) if val < SAFE_YEAR else ">2075"

arch_short = ["A: Low (1.0 ft)", "B: Mid (2.5 ft)", "C: High (4.0 ft)"]

header  = f"{'':22s}" + "".join(f"{sc:>16s}" for sc in scenario_keys)
divider = "-" * (22 + 16 * len(scenario_keys))

print("\nT_c — Year of Chronic Inundation Onset")
print(divider)
print(header)
print(divider)
for short, label in zip(arch_short, arch_labels):
    row = f"{short:<22s}" + "".join(
        f"{tc_str(tc[label][sc]):>16s}" for sc in scenario_keys
    )
    print(row)
print(divider)
print()

# ---------------------------------------------------------------------------
# 3×3 subplot grid
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)

fig, axes = plt.subplots(
    3, 3,
    figsize=(15, 11),
    sharex=True,
    sharey="row",
    constrained_layout=True,
)

for col_idx, (sc, sc_col) in enumerate(SCENARIOS.items()):
    color = SCENARIO_COLORS[sc]

    for row_idx, (label, elev) in enumerate(ARCHETYPES.items()):
        ax  = axes[row_idx, col_idx]
        d   = depths[label][sc]
        t_c = tc[label][sc]

        # --- Shaded fill ------------------------------------------------
        ax.fill_between(years, d, 0,
                        where=(d >= 0),
                        color="#FFCDD2", alpha=0.75,
                        label="Flooded (d > 0)")
        ax.fill_between(years, d, 0,
                        where=(d <= 0),
                        color="#BBDEFB", alpha=0.75,
                        label="Dry (d < 0)")

        # --- Depth curve ------------------------------------------------
        ax.plot(years, d, color=color, linewidth=2.0, zorder=3)

        # --- d = 0 reference line ---------------------------------------
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9, zorder=2)

        # --- T_c marker -------------------------------------------------
        if t_c < SAFE_YEAR:
            ax.axvline(t_c, color="#E65100", linewidth=1.4,
                       linestyle="-", zorder=4)
            # label above x-axis, inside plot
            ax.text(t_c + 0.4, ax.get_ylim()[0] * 0.85 if ax.get_ylim()[0] < 0 else 0.1,
                    f"Tс={t_c}", color="#E65100",
                    fontsize=7.5, va="bottom", rotation=90,
                    transform=ax.get_xaxis_transform())
        else:
            ax.text(0.97, 0.05, "No chronic\ninundation",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="#1B5E20",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        # --- Titles and labels ------------------------------------------
        short_label = arch_short[row_idx]
        ax.set_title(f"{short_label} | {sc}", fontsize=9, pad=4)

        if col_idx == 0:
            ax.set_ylabel("Inundation Depth (ft)", fontsize=8)
        if row_idx == 2:
            ax.set_xlabel("Year", fontsize=8)

        ax.set_xlim(2025, 2075)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25)

# --- Shared legend ------------------------------------------------------
patch_flood = mpatches.Patch(color="#FFCDD2", label="Flooded (d > 0)")
patch_dry   = mpatches.Patch(color="#BBDEFB", label="Dry (d < 0)")
line_zero   = mlines.Line2D([], [], color="black", linestyle="--",
                             linewidth=0.9, label="d = 0 (BFE line)")
line_tc     = mlines.Line2D([], [], color="#E65100", linewidth=1.4,
                             label="T_c onset")

fig.legend(
    handles=[patch_flood, patch_dry, line_zero, line_tc],
    loc="lower center",
    ncol=4,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.03),
)

fig.suptitle(
    "100-Year Flood Inundation Depth Over Time\n"
    f"BFE = {BFE_FT} ft NAVD88, MHHW = {MHHW_FT} ft NAVD88, Brickell Study Area",
    fontsize=13,
    fontweight="bold",
)

fig.savefig(str(PLOT_OUT), dpi=200, bbox_inches="tight")
print(f"Saved plot → {PLOT_OUT}")
