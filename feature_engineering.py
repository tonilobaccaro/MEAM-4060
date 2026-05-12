"""
Feature engineering for the parcel-level flood-risk classifier.

Inputs  : data/study_parcels_physics.gpkg
Outputs : outputs/X_scaled.npy, outputs/y_labels.npy,
          outputs/feature_names.txt, outputs/feature_scaler.pkl,
          outputs/feature_correlation.png
"""

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARCELS_IN   = pathlib.Path("data/study_parcels_physics.gpkg")
SCALER_OUT   = pathlib.Path("outputs/feature_scaler.pkl")
X_OUT        = pathlib.Path("outputs/X_scaled.npy")
Y_OUT        = pathlib.Path("outputs/y_labels.npy")
NAMES_OUT    = pathlib.Path("outputs/feature_names.txt")
CORR_OUT     = pathlib.Path("outputs/feature_correlation.png")

TARGET_CRS   = "EPSG:6437"
LABEL_COL    = "T_c_intermediate"
LABEL_CUTOFF = 2050

# Biscayne Bay shoreline approximated as a vertical line at lon -80.187.
# We define two points spanning the study-area latitude range.
SHORELINE_WGS84 = [(-80.187, 25.750), (-80.187, 25.780)]

# ---------------------------------------------------------------------------
# 2. Desired feature columns (in order)
# ---------------------------------------------------------------------------
DESIRED_FEATURES = [
    "elev_ft_navd88",
    "bfe_ft",
    "freeboard_ft",
    "d_100yr_2025",
    "d_100yr_2050_int",
    "fld_zone_risk",
]

# ---------------------------------------------------------------------------
# 1. Load parcels
# ---------------------------------------------------------------------------
if not PARCELS_IN.exists():
    print(f"ERROR: {PARCELS_IN} not found — run inundation_model.py first.")
    sys.exit(1)

print(f"Loading {PARCELS_IN} …")
gdf = gpd.read_file(str(PARCELS_IN)).to_crs(TARGET_CRS)
print(f"  {len(gdf):,} parcels loaded\n")

# ---------------------------------------------------------------------------
# 2. Select features (graceful handling of missing columns)
# ---------------------------------------------------------------------------
present  = [c for c in DESIRED_FEATURES if c in gdf.columns]
missing  = [c for c in DESIRED_FEATURES if c not in gdf.columns]

if missing:
    print(f"WARNING: The following feature columns were not found and will be "
          f"omitted:\n  {missing}\n")

feature_cols = present
print(f"Features selected ({len(feature_cols)}):")
for fc in feature_cols:
    print(f"  {fc}")
print()

# Sanity check — prevent accidental leakage columns
LEAKAGE_COLS = {"T_c_low", "T_c_intermediate", "T_c_high",
                "chronic_by_2050_intermediate"}
leakage_found = LEAKAGE_COLS & set(feature_cols)
if leakage_found:
    raise RuntimeError(f"Leakage columns detected in features: {leakage_found}")

# ---------------------------------------------------------------------------
# 3. Binary label
# ---------------------------------------------------------------------------
if LABEL_COL not in gdf.columns:
    print(f"ERROR: Label column '{LABEL_COL}' not found — run inundation_model.py.")
    sys.exit(1)

y_series = (gdf[LABEL_COL] <= LABEL_CUTOFF).astype(int)
n1 = y_series.sum()
n0 = len(y_series) - n1
print(f"Class balance  (label = T_c_intermediate ≤ {LABEL_CUTOFF}):")
print(f"  y=1 (high near-term risk) : {n1:,}  ({100*n1/len(y_series):.1f}%)")
print(f"  y=0 (lower near-term risk): {n0:,}  ({100*n0/len(y_series):.1f}%)\n")

# ---------------------------------------------------------------------------
# 4. Distance-to-coast feature
# ---------------------------------------------------------------------------
print("Computing dist_to_coast_m …")

# Build the shoreline in WGS84 then reproject to EPSG:6437
from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
shore_pts_proj = [transformer.transform(lon, lat) for lon, lat in SHORELINE_WGS84]
shoreline = LineString(shore_pts_proj)

centroids = gdf.geometry.centroid   # already in EPSG:6437 (metric)
gdf["dist_to_coast_m"] = centroids.distance(shoreline)

print(f"  min={gdf['dist_to_coast_m'].min():.1f} m  "
      f"max={gdf['dist_to_coast_m'].max():.1f} m  "
      f"mean={gdf['dist_to_coast_m'].mean():.1f} m\n")

feature_cols = feature_cols + ["dist_to_coast_m"]

# ---------------------------------------------------------------------------
# 5. Drop NaN rows
# ---------------------------------------------------------------------------
df_feat = gdf[feature_cols + [LABEL_COL]].copy()
n_before = len(df_feat)
df_feat = df_feat.dropna(subset=feature_cols)
n_dropped = n_before - len(df_feat)

if n_dropped:
    print(f"Dropped {n_dropped:,} rows with NaN in feature columns.")
else:
    print("No rows dropped (no NaN in feature columns).")

y_series = (df_feat[LABEL_COL] <= LABEL_CUTOFF).astype(int)
X_raw = df_feat[feature_cols].values
y     = y_series.values
print(f"Final dataset: {X_raw.shape[0]:,} samples × {X_raw.shape[1]} features\n")

# ---------------------------------------------------------------------------
# 6. Standardize
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

joblib.dump(scaler, str(SCALER_OUT))
print(f"Scaler saved → {SCALER_OUT}")

# ---------------------------------------------------------------------------
# 7. Save arrays and feature names
# ---------------------------------------------------------------------------
np.save(str(X_OUT), X_scaled)
np.save(str(Y_OUT), y)

NAMES_OUT.write_text("\n".join(feature_cols) + "\n")

print(f"Saved → {X_OUT}")
print(f"Saved → {Y_OUT}")
print(f"Saved → {NAMES_OUT}\n")

# ---------------------------------------------------------------------------
# 8. Shape and feature name report
# ---------------------------------------------------------------------------
print(f"X_scaled shape : {X_scaled.shape}")
print("Feature names  :")
for i, name in enumerate(feature_cols):
    mean_raw = float(scaler.mean_[i])
    std_raw  = float(scaler.scale_[i])
    print(f"  [{i}] {name:<25s}  μ={mean_raw:+.3f}  σ={std_raw:.3f}")
print()

# ---------------------------------------------------------------------------
# 9. Correlation matrix heatmap
# ---------------------------------------------------------------------------
df_corr = pd.DataFrame(X_raw, columns=feature_cols).corr()

fig, ax = plt.subplots(figsize=(9, 7))

# Symmetric diverging colormap centred at 0
vmax = 1.0
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im   = ax.imshow(df_corr.values, cmap="RdBu_r", norm=norm, aspect="auto")

# Axis labels
n = len(feature_cols)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(feature_cols, fontsize=9)

# Cell annotations
for i in range(n):
    for j in range(n):
        val = df_corr.iloc[i, j]
        txt_color = "white" if abs(val) > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=txt_color)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
ax.set_title("Feature Correlation Matrix — Flood Risk Model",
             fontsize=12, fontweight="bold", pad=12)

fig.tight_layout()
fig.savefig(str(CORR_OUT), dpi=200)
print(f"Saved plot → {CORR_OUT}")
