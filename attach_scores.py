"""
Attach ML risk scores back to the spatial GeoDataFrame and export final outputs.

Inputs  : data/study_parcels_physics.gpkg
          outputs/risk_scores.npy
          outputs/y_labels.npy
          outputs/feature_names.txt
Outputs : data/study_parcels_scored.gpkg        (EPSG:6437)
          data/study_parcels_scored_wgs84.gpkg   (EPSG:4326)
          outputs/top_20_risk_parcels.csv
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARCELS_IN    = pathlib.Path("data/study_parcels_physics.gpkg")
SCORES_PATH   = pathlib.Path("outputs/risk_scores.npy")
LABELS_PATH   = pathlib.Path("outputs/y_labels.npy")
NAMES_PATH    = pathlib.Path("outputs/feature_names.txt")

OUT_SCORED    = pathlib.Path("data/study_parcels_scored.gpkg")
OUT_WGS84     = pathlib.Path("data/study_parcels_scored_wgs84.gpkg")
OUT_TOP20     = pathlib.Path("outputs/top_20_risk_parcels.csv")

TARGET_CRS    = "EPSG:6437"
WGS84_CRS     = "EPSG:4326"

LOW_MAX   = 33
HIGH_MIN  = 67

# Columns to show in top-20 table (checked for existence before use)
TOP20_COLS = [
    "parcel_id", "elev_ft_navd88", "fld_zone",
    "T_c_low", "T_c_intermediate", "T_c_high",
    "risk_score", "risk_category",
]

# ---------------------------------------------------------------------------
# 1. Load parcels
# ---------------------------------------------------------------------------
for p in (PARCELS_IN, SCORES_PATH, LABELS_PATH, NAMES_PATH):
    if not p.exists():
        prereq = {
            PARCELS_IN:  "inundation_model.py",
            SCORES_PATH: "train_model.py",
            LABELS_PATH: "train_model.py",
            NAMES_PATH:  "feature_engineering.py",
        }[p]
        print(f"ERROR: {p} not found — run {prereq} first.")
        sys.exit(1)

print(f"Loading {PARCELS_IN} …")
gdf = gpd.read_file(str(PARCELS_IN)).to_crs(TARGET_CRS)
print(f"  {len(gdf):,} parcels loaded")

# ---------------------------------------------------------------------------
# 2-3. Load arrays and confirm alignment
# ---------------------------------------------------------------------------
risk_scores   = np.load(str(SCORES_PATH))
y_labels      = np.load(str(LABELS_PATH))
feature_names = NAMES_PATH.read_text().strip().splitlines()

print(f"  risk_scores  : {risk_scores.shape}")
print(f"  y_labels     : {y_labels.shape}")
print(f"  feature_names: {feature_names}\n")

# The feature matrix was built from the parcel GDF after dropping NaN rows,
# so its length may differ from the full GDF. Detect and report.
if len(risk_scores) != len(gdf):
    print(
        f"WARNING: risk_scores length ({len(risk_scores):,}) != GDF length "
        f"({len(gdf):,}). The score array was computed on a NaN-dropped subset.\n"
        f"  Scores will be assigned to the first {len(risk_scores):,} rows; "
        f"remaining rows will receive NaN risk_score."
    )

# ---------------------------------------------------------------------------
# 4. Attach columns
# ---------------------------------------------------------------------------
def categorise(score):
    if pd.isna(score):
        return "Unknown"
    if score < LOW_MAX:
        return "Low"
    if score <= HIGH_MIN:
        return "Medium"
    return "High"

# Pre-fill with NaN so any unscored rows get NaN rather than garbage
gdf["risk_score"]      = np.nan
gdf["high_risk_label"] = np.nan

n_scored = min(len(risk_scores), len(gdf))
gdf.iloc[:n_scored, gdf.columns.get_loc("risk_score")]      = risk_scores[:n_scored]
gdf.iloc[:n_scored, gdf.columns.get_loc("high_risk_label")] = y_labels[:n_scored]

gdf["high_risk_label"] = gdf["high_risk_label"].astype("Int64")   # nullable int
gdf["risk_category"]   = gdf["risk_score"].apply(categorise)

print("Columns attached: risk_score, risk_category, high_risk_label")
print(f"  Scored parcels : {n_scored:,}")
print(f"  Unscored (NaN) : {len(gdf) - n_scored:,}\n")

# ---------------------------------------------------------------------------
# 5. Summary table grouped by risk_category
# ---------------------------------------------------------------------------
scored = gdf[gdf["risk_score"].notna()]

summary_agg = {}
for col in ["elev_ft_navd88", "T_c_intermediate", "risk_score"]:
    if col in scored.columns:
        summary_agg[col] = "mean"

summary = (
    scored.groupby("risk_category")
          .agg(parcels=("risk_score", "count"), **{
              k: (k, v) for k, v in summary_agg.items()
          })
          .rename(columns={
              "elev_ft_navd88":  "mean_elev_ft",
              "T_c_intermediate": "mean_Tc_int",
              "risk_score":       "mean_risk_score",
          })
          .round(2)
)

# Re-order rows by risk level
cat_order = ["Low", "Medium", "High", "Unknown"]
summary = summary.reindex([c for c in cat_order if c in summary.index])

print("--- Risk Category Summary ---")
print(summary.to_string())
print()

# ---------------------------------------------------------------------------
# 6. Top 20 highest-risk parcels
# ---------------------------------------------------------------------------
avail_cols = [c for c in TOP20_COLS if c in gdf.columns]
missing_top20 = set(TOP20_COLS) - set(avail_cols)
if missing_top20:
    print(f"Note: top-20 table omits missing columns: {sorted(missing_top20)}")

top20 = (
    gdf[gdf["risk_score"].notna()]
    .nlargest(20, "risk_score")[avail_cols]
    .reset_index(drop=True)
)
top20.index += 1                     # rank 1-20

print("--- Top 20 Highest-Risk Parcels ---")
print(top20.to_string())
print()

pathlib.Path("outputs").mkdir(exist_ok=True)
top20.to_csv(str(OUT_TOP20), index_label="rank")
print(f"Saved → {OUT_TOP20}\n")

# ---------------------------------------------------------------------------
# 7. Save scored GeoDataFrame (EPSG:6437)
# ---------------------------------------------------------------------------
pathlib.Path("data").mkdir(exist_ok=True)
gdf.to_file(str(OUT_SCORED), driver="GPKG")
print(f"Saved → {OUT_SCORED}  ({len(gdf):,} features, {TARGET_CRS})")

# ---------------------------------------------------------------------------
# 8. WGS84 version for folium
# ---------------------------------------------------------------------------
gdf_wgs84 = gdf.to_crs(WGS84_CRS)
gdf_wgs84.to_file(str(OUT_WGS84), driver="GPKG")
print(f"Saved → {OUT_WGS84}  ({len(gdf_wgs84):,} features, {WGS84_CRS})")
