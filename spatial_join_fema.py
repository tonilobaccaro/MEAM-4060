"""
Spatial join: attach FEMA flood zones and BFE to study-area parcels.
Resolves multi-zone duplicates by risk priority, fills missing BFE from
nearest BFE lines then zone-median fallback, computes freeboard.
"""

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARCELS_PATH    = pathlib.Path("data/study_parcels_elev.gpkg")
ZONES_PATH      = pathlib.Path("data/fema_flood_zones.gpkg")
BFE_LINES_PATH  = pathlib.Path("data/fema_bfe_lines.gpkg")
OUT_PATH        = pathlib.Path("data/study_parcels_full.gpkg")

TARGET_CRS      = "EPSG:6437"
BFE_MAX_DIST    = 500        # metres — max distance for nearest-BFE-line search

# Risk priority: higher number = higher risk
ZONE_PRIORITY = {"VE": 6, "AE": 5, "AH": 4, "AO": 3, "X_500": 2, "X": 1}

# Integer risk level for fld_zone_risk column
ZONE_RISK_LEVEL = {"VE": 4, "AE": 3, "AH": 2, "AO": 2, "X_500": 1, "X": 0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(path: pathlib.Path, label: str) -> gpd.GeoDataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found — run the prerequisite script first.")
        sys.exit(1)
    gdf = gpd.read_file(str(path)).to_crs(TARGET_CRS)
    print(f"  {label}: {len(gdf):,} features")
    return gdf


def _zone_key(zone_str) -> str:
    """Normalise a raw FLD_ZONE string to a canonical key."""
    z = str(zone_str).strip().upper()
    if z.startswith("X") and ("500" in z or "SHADED" in z):
        return "X_500"
    if z.startswith("X"):
        return "X"
    return z


def _priority(zone_str) -> int:
    return ZONE_PRIORITY.get(_zone_key(zone_str), 0)


# ---------------------------------------------------------------------------
# 1-3. Load layers
# ---------------------------------------------------------------------------
print("Loading layers …")
parcels   = _load(PARCELS_PATH,   "Parcels (with elevation)")
zones     = _load(ZONES_PATH,     "FEMA flood zones")
bfe_lines = _load(BFE_LINES_PATH, "BFE lines")
print()

# Keep only needed zone columns (guard against missing columns gracefully)
zone_keep = [c for c in ["FLD_ZONE", "STATIC_BFE", "geometry"] if c in zones.columns]
missing_zone_cols = {"FLD_ZONE", "STATIC_BFE"} - set(zones.columns)
if missing_zone_cols:
    print(f"WARNING: Zone layer missing columns {missing_zone_cols}; will work with what's available.")
zones_slim = zones[zone_keep].copy()

# ---------------------------------------------------------------------------
# 4. Spatial join: parcels within flood zones
# ---------------------------------------------------------------------------
print("Spatial join: parcels within flood zones …")

# Use centroid for within-predicate (parcel polygons can straddle zone boundaries)
parcels_c = parcels.copy()
parcels_c["_orig_geom"] = parcels_c.geometry
parcels_c.geometry = parcels_c.geometry.centroid

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # suppress index-name collision warning
    joined = gpd.sjoin(
        parcels_c,
        zones_slim,
        how="left",
        predicate="within",
    )

# Restore polygon geometry
joined.geometry = joined["_orig_geom"]
joined = joined.drop(columns=["_orig_geom", "index_right"], errors="ignore")

# Resolve duplicates: keep highest-risk zone per parcel
if joined.index.duplicated().any():
    joined["_prio"] = joined["FLD_ZONE"].apply(_priority)
    joined = (
        joined.sort_values("_prio", ascending=False)
              .loc[~joined.index.duplicated(keep="first")]
    )
    joined = joined.drop(columns=["_prio"])

print(f"  Parcels after zone join: {len(joined):,}")
print(f"  Zone distribution:\n{joined['FLD_ZONE'].value_counts().to_string()}\n")

# Rename STATIC_BFE → bfe_ft as our working BFE column
if "STATIC_BFE" in joined.columns:
    joined = joined.rename(columns={"STATIC_BFE": "bfe_ft"})
    # Treat 0 and negative as missing (FEMA uses 0 for unmapped areas)
    joined["bfe_ft"] = pd.to_numeric(joined["bfe_ft"], errors="coerce")
    joined.loc[joined["bfe_ft"] <= 0, "bfe_ft"] = np.nan
else:
    joined["bfe_ft"] = np.nan

# ---------------------------------------------------------------------------
# 5. Fill missing BFE from nearest BFE line (≤ 500 m)
# ---------------------------------------------------------------------------
bfe_col_in_lines = "BFE_LEN" if "BFE_LEN" in bfe_lines.columns else None
if bfe_col_in_lines is None:
    print("WARNING: BFE_LEN column not found in BFE lines — skipping line-based fill.")
    n_from_lines = 0
else:
    need_bfe_mask = joined["bfe_ft"].isna()
    n_need = need_bfe_mask.sum()
    print(f"Parcels needing BFE from lines: {n_need:,}")

    if n_need > 0:
        parcels_need = joined.loc[need_bfe_mask, ["geometry"]].copy()
        parcels_need.geometry = parcels_need.geometry.centroid

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn = gpd.sjoin_nearest(
                parcels_need,
                bfe_lines[["geometry", bfe_col_in_lines]].copy(),
                how="left",
                max_distance=BFE_MAX_DIST,
                distance_col="_dist",
            )

        # sjoin_nearest may produce duplicates if equidistant — keep closest
        nn = nn[~nn.index.duplicated(keep="first")]
        nn_bfe = pd.to_numeric(nn[bfe_col_in_lines], errors="coerce")
        nn_bfe = nn_bfe[nn_bfe > 0]

        joined.loc[nn_bfe.index, "bfe_ft"] = nn_bfe
        n_from_lines = (~nn_bfe.isna()).sum()
        print(f"  Parcels assigned BFE from lines: {n_from_lines:,}\n")
    else:
        n_from_lines = 0
        print("  No parcels needed BFE from lines.\n")

# ---------------------------------------------------------------------------
# 6. Fill remaining missing BFE with zone-median fallback
# ---------------------------------------------------------------------------
still_missing = joined["bfe_ft"].isna()
n_still = still_missing.sum()

if n_still > 0:
    zone_median_bfe = joined.groupby("FLD_ZONE")["bfe_ft"].median()
    zone_filled = 0
    for zone, med in zone_median_bfe.items():
        if pd.isna(med):
            continue
        mask = still_missing & (joined["FLD_ZONE"] == zone)
        joined.loc[mask, "bfe_ft"] = med
        zone_filled += mask.sum()
    print(f"Parcels filled with zone-median BFE: {zone_filled:,}")
else:
    print("No parcels needed zone-median BFE fill.")

# ---------------------------------------------------------------------------
# 7. Flood-zone risk integer
# ---------------------------------------------------------------------------
joined["fld_zone_risk"] = joined["FLD_ZONE"].apply(
    lambda z: ZONE_RISK_LEVEL.get(_zone_key(z), 0)
)

# Canonical zone name (strips internal whitespace/case variants)
joined["fld_zone"] = joined["FLD_ZONE"].apply(_zone_key)

# ---------------------------------------------------------------------------
# 8. Freeboard
# ---------------------------------------------------------------------------
joined["freeboard_ft"] = joined["elev_ft_navd88"] - joined["bfe_ft"]

n_below = (joined["freeboard_ft"] < 0).sum()
print(f"\nParcels below BFE (negative freeboard): {n_below:,}")

# ---------------------------------------------------------------------------
# 9. Summary table by flood zone
# ---------------------------------------------------------------------------
summary_cols = {
    "parcel_id":      "count",
    "elev_ft_navd88": "mean",
    "bfe_ft":         "mean",
    "freeboard_ft":   "mean",
}
summary = (
    joined.groupby("fld_zone")
          .agg(**{
              "parcels":       ("parcel_id",      "count"),
              "mean_elev_ft":  ("elev_ft_navd88", "mean"),
              "mean_bfe_ft":   ("bfe_ft",          "mean"),
              "mean_freeboard":("freeboard_ft",    "mean"),
          })
          .round(3)
          .sort_values("mean_freeboard")
)

print("\n--- Flood Zone Summary ---")
print(summary.to_string())
print()

# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
out_cols = list(dict.fromkeys(
    list(parcels.columns) +
    ["fld_zone", "bfe_ft", "fld_zone_risk", "freeboard_ft"]
))
# Only keep columns that actually exist in joined
out_cols = [c for c in out_cols if c in joined.columns]

pathlib.Path("data").mkdir(exist_ok=True)
joined[out_cols].to_file(str(OUT_PATH), driver="GPKG")
print(f"Saved → {OUT_PATH}  ({len(joined):,} features)")
print(f"  Output columns: {[c for c in out_cols if c != 'geometry']}")
