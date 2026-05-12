"""
2×2 summary panel for the Brickell SLR flood-risk assessment.

Sources:
  data/slr_scenarios.csv
  data/study_parcels_physics.gpkg
  outputs/lr_model.pkl
  outputs/feature_names.txt
  outputs/risk_scores.npy
"""

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SLR_CSV       = pathlib.Path("data/slr_scenarios.csv")
PARCELS_PATH  = pathlib.Path("data/study_parcels_physics.gpkg")
MODEL_PATH    = pathlib.Path("outputs/lr_model.pkl")
NAMES_PATH    = pathlib.Path("outputs/feature_names.txt")
SCORES_PATH   = pathlib.Path("outputs/risk_scores.npy")
PANEL_OUT     = pathlib.Path("outputs/summary_panel.png")

SAFE_YEAR     = 2076
MHHW_FT       = 0.87

# ---------------------------------------------------------------------------
# Check required inputs
# ---------------------------------------------------------------------------
required = {
    SLR_CSV:      "slr_scenarios.py",
    PARCELS_PATH: "inundation_model.py",
    MODEL_PATH:   "train_model.py",
    NAMES_PATH:   "feature_engineering.py",
    SCORES_PATH:  "train_model.py",
}
missing = [p for p in required if not p.exists()]
if missing:
    for p in missing:
        print(f"ERROR: {p} not found — run {required[p]} first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
slr        = pd.read_csv(str(SLR_CSV)).set_index("year")
gdf        = gpd.read_file(str(PARCELS_PATH))
model      = joblib.load(str(MODEL_PATH))
feat_names = NAMES_PATH.read_text().strip().splitlines()
risk_scores = np.load(str(SCORES_PATH))

print(f"Loaded: {len(gdf):,} parcels  |  {len(feat_names)} features  |  "
      f"{len(risk_scores):,} risk scores")

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10),
    gridspec_kw={"hspace": 0.38, "wspace": 0.32},
)
ax_slr, ax_tc, ax_imp, ax_risk = axes.flat

PANEL_LABEL_KW = dict(
    fontsize=11, fontweight="bold",
    transform=None,   # set per-axis below
    va="top", ha="left",
    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1),
)

def add_panel_label(ax, letter, text):
    ax.text(
        0.01, 0.99, f"{letter}. {text}",
        transform=ax.transAxes,
        fontsize=9.5, fontweight="bold",
        va="top", ha="left",
        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1),
    )

# ===========================================================================
# A — SLR Scenarios
# ===========================================================================
SCENARIO_STYLE = {
    "slr_low_ft":          {"label": "Low",         "color": "#2196F3", "lw": 1.8},
    "slr_intermediate_ft": {"label": "Intermediate", "color": "#FF9800", "lw": 1.8},
    "slr_high_ft":         {"label": "High",         "color": "#F44336", "lw": 1.8},
}

years = slr.index.values
for col, style in SCENARIO_STYLE.items():
    ax_slr.plot(years, slr[col], color=style["color"],
                linewidth=style["lw"], label=style["label"])

ax_slr.axvline(2050, color="gray", linestyle="--", linewidth=0.8)
ax_slr.axhline(0,    color="black", linestyle="--", linewidth=0.7)

# 2050 value labels
row_2050 = slr.loc[2050]
for col, style in SCENARIO_STYLE.items():
    val = row_2050[col]
    ax_slr.annotate(
        f"{val:.1f} ft",
        xy=(2050, val), xytext=(4, 0), textcoords="offset points",
        va="center", fontsize=7.5, color=style["color"],
    )

ax_slr.set_xlim(2025, 2075)
ax_slr.set_ylim(bottom=-0.05)
ax_slr.set_xlabel("Year", fontsize=9)
ax_slr.set_ylabel("SLR above 2000 baseline (ft)", fontsize=9)
ax_slr.legend(fontsize=8, loc="upper left")
ax_slr.grid(axis="y", alpha=0.25)
ax_slr.tick_params(labelsize=8)
add_panel_label(ax_slr, "A", "NOAA SLR Scenarios — Virginia Key")

# ===========================================================================
# B — T_c histogram (Intermediate only)
# ===========================================================================
TC_COL = "T_c_intermediate"

if TC_COL in gdf.columns:
    tc_vals = gdf[TC_COL].values.astype(float)
else:
    tc_vals = np.full(len(gdf), SAFE_YEAR, dtype=float)
    print(f"WARNING: {TC_COL} not found — filling with SAFE_YEAR.")

# Decade bins
BINS   = [2025, 2035, 2045, 2055, 2065, 2076, 2077]   # 2076+ bucket for safe
LABELS = ["2025–35", "2035–45", "2045–55", "2055–65", "2065–75", ">2075"]
BCOLORS = ["#8B0000", "#FF6600", "#FFD700", "#9ACD32", "#228B22", "#AAAAAA"]

counts = []
for lo, hi in zip(BINS[:-1], BINS[1:]):
    counts.append(int(np.sum((tc_vals >= lo) & (tc_vals < hi))))

# Extend safe-year bucket to catch anything ≥ 2076
counts[-1] = int(np.sum(tc_vals >= SAFE_YEAR))

x_pos = np.arange(len(LABELS))
bars  = ax_tc.bar(x_pos, counts, color=BCOLORS, edgecolor="white", linewidth=0.5)

ax_tc.set_xticks(x_pos)
ax_tc.set_xticklabels(LABELS, fontsize=7.5, rotation=20, ha="right")
ax_tc.set_ylabel("Number of Parcels", fontsize=9)
ax_tc.set_xlabel("Year of Chronic Inundation Onset", fontsize=9)
ax_tc.tick_params(labelsize=8)
ax_tc.grid(axis="y", alpha=0.25)

# Value labels on bars
for bar, count in zip(bars, counts):
    if count > 0:
        ax_tc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center", va="bottom", fontsize=7.5,
        )

add_panel_label(ax_tc, "B", "Chronic Inundation Onset (Intermediate)")

# ===========================================================================
# C — Feature importance (standardized LR coefficients)
# ===========================================================================
coefs  = model.coef_[0]
order  = np.argsort(np.abs(coefs))          # ascending → most important at top
sorted_names = [feat_names[i] for i in order]
sorted_coefs = coefs[order]
bar_colors   = ["#F44336" if c > 0 else "#2196F3" for c in sorted_coefs]

bars_h = ax_imp.barh(
    sorted_names, sorted_coefs,
    color=bar_colors, edgecolor="white", height=0.6,
)

# Value labels
for bar, val in zip(bars_h, sorted_coefs):
    offset = 0.03 if val >= 0 else -0.03
    ha     = "left" if val >= 0 else "right"
    ax_imp.text(
        val + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}",
        va="center", ha=ha, fontsize=7,
    )

ax_imp.axvline(0, color="black", linewidth=0.8)
ax_imp.set_xlabel("Standardized Coefficient", fontsize=9)
ax_imp.tick_params(labelsize=7.5)
ax_imp.grid(axis="x", alpha=0.2)

red_p  = mpatches.Patch(color="#F44336", label="↑ risk")
blue_p = mpatches.Patch(color="#2196F3", label="↓ risk")
ax_imp.legend(handles=[red_p, blue_p], fontsize=7.5, loc="lower right")
add_panel_label(ax_imp, "C", "Risk Score Feature Importances")

# ===========================================================================
# D — Risk score distribution
# ===========================================================================
BINS_RISK  = np.linspace(0, 100, 21)
bin_centers = (BINS_RISK[:-1] + BINS_RISK[1:]) / 2
hist_counts, _ = np.histogram(risk_scores, bins=BINS_RISK)

bar_risk_colors = [
    "#4CAF50" if c < 33 else "#FFC107" if c < 67 else "#F44336"
    for c in bin_centers
]

ax_risk.bar(
    bin_centers, hist_counts,
    width=5.0, color=bar_risk_colors,
    edgecolor="white", linewidth=0.5, align="center",
)

for thr, label in [(33, "Low / Med"), (67, "Med / High")]:
    ax_risk.axvline(thr, color="black", linestyle="--", linewidth=0.9)
    ax_risk.text(
        thr + 0.8,
        ax_risk.get_ylim()[1] * 0.96,
        label, fontsize=7, va="top",
    )

n_low  = int((risk_scores  < 33).sum())
n_med  = int(((risk_scores >= 33) & (risk_scores <= 67)).sum())
n_high = int((risk_scores  > 67).sum())

green_p  = mpatches.Patch(color="#4CAF50", label=f"Low (<33)  n={n_low}")
yellow_p = mpatches.Patch(color="#FFC107", label=f"Med (33-67) n={n_med}")
red_p2   = mpatches.Patch(color="#F44336", label=f"High (>67) n={n_high}")
ax_risk.legend(handles=[green_p, yellow_p, red_p2], fontsize=7.5, loc="upper center")

ax_risk.set_xlabel("Risk Score (0 = lowest, 100 = highest)", fontsize=9)
ax_risk.set_ylabel("Number of Parcels", fontsize=9)
ax_risk.set_xlim(0, 100)
ax_risk.tick_params(labelsize=8)
ax_risk.grid(axis="y", alpha=0.25)
add_panel_label(ax_risk, "D", "Flood Risk Score Distribution")

# ===========================================================================
# Overall title and save
# ===========================================================================
fig.suptitle(
    "Sea-Level-Rise Flood Risk Assessment — Brickell Waterfront, Miami",
    fontsize=13, fontweight="bold", y=0.995,
)

pathlib.Path("outputs").mkdir(exist_ok=True)
fig.savefig(str(PANEL_OUT), dpi=300, bbox_inches="tight")
print(f"\nSaved → {PANEL_OUT}  (300 DPI, 14×10 in)")
