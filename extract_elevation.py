"""
Sample DEM elevation at each study-area parcel centroid, fill NaN gaps with
nearest-neighbour median, and save/plot results.
"""

import pathlib
import sys
import warnings

import numpy as np
import geopandas as gpd
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PARCELS_IN  = pathlib.Path("data/study_parcels.gpkg")
DEM_PATH    = pathlib.Path("data/dem_brickell.tif")
PARCELS_OUT = pathlib.Path("data/study_parcels_elev.gpkg")
PLOT_OUT    = pathlib.Path("outputs/elevation_histogram.png")

PARCEL_CRS  = "EPSG:6437"
MHHW_FT     = 0.87
M_TO_FT     = 3.28084
KNN         = 5          # neighbours for NaN fill

VLINES = [
    (MHHW_FT, "MHHW\n0.87 ft", "#F44336"),
    (2.00,    "2.00 ft",        "#FF9800"),
    (3.00,    "3.00 ft",        "#9C27B0"),
]

# ---------------------------------------------------------------------------
# 1. Load parcels
# ---------------------------------------------------------------------------
if not PARCELS_IN.exists():
    print(f"ERROR: {PARCELS_IN} not found — run load_parcels.py first.")
    sys.exit(1)

print(f"Loading {PARCELS_IN} …")
gdf = gpd.read_file(str(PARCELS_IN)).to_crs(PARCEL_CRS)
print(f"  {len(gdf):,} parcels loaded  (CRS: {gdf.crs})\n")

# ---------------------------------------------------------------------------
# 2. Open DEM
# ---------------------------------------------------------------------------
if not DEM_PATH.exists():
    print(
        "ERROR: DEM not found.\n"
        "  Download the 1/3 arc-second DEM for the study area from\n"
        "  https://apps.nationalmap.gov/downloader\n"
        f"  and save as {DEM_PATH}"
    )
    sys.exit(1)

print(f"Opening DEM: {DEM_PATH}")
with rasterio.open(str(DEM_PATH)) as src:
    dem_crs    = src.crs
    dem_res    = src.res                         # (x_res, y_res) in CRS units
    dem_bounds = src.bounds
    dem_nodata = src.nodata
    dem_transform = src.transform

    print(f"  CRS      : {dem_crs}")
    print(f"  Resolution: {dem_res[0]:.6f} x {dem_res[1]:.6f}  ({dem_crs.linear_units if not dem_crs.is_geographic else 'degrees'})")
    print(f"  Bounds   : left={dem_bounds.left:.4f}  right={dem_bounds.right:.4f}")
    print(f"             bottom={dem_bounds.bottom:.4f}  top={dem_bounds.top:.4f}")
    print(f"  NoData   : {dem_nodata}\n")

    # -----------------------------------------------------------------------
    # 3. Parcel centroids → DEM CRS
    # -----------------------------------------------------------------------
    centroids = gdf.geometry.centroid             # still in PARCEL_CRS

    # Build transformer from parcel CRS → DEM CRS
    transformer = Transformer.from_crs(
        PARCEL_CRS, str(dem_crs), always_xy=True
    )
    cx_dem, cy_dem = transformer.transform(
        centroids.x.values, centroids.y.values
    )
    coord_pairs = list(zip(cx_dem, cy_dem))

    # -----------------------------------------------------------------------
    # 4. Sample DEM
    # -----------------------------------------------------------------------
    sampled = np.array(
        [v[0] for v in src.sample(coord_pairs, indexes=1)],
        dtype=float,
    )

# Replace nodata with NaN
if dem_nodata is not None:
    sampled[np.isclose(sampled, dem_nodata, equal_nan=True)] = np.nan

# Guard: values that rasterio returns for out-of-bounds coords are typically
# nodata or 0; treat extreme outliers as NaN too.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    sampled[sampled < -999] = np.nan

# ---------------------------------------------------------------------------
# 5. Unit detection — convert meters → feet if needed
# ---------------------------------------------------------------------------
valid = sampled[~np.isnan(sampled)]

if len(valid) == 0:
    print("WARNING: All sampled elevations are NaN — check DEM coverage.")
else:
    med = float(np.median(valid))
    if 0 <= med <= 10:
        sampled = sampled * M_TO_FT
        print("Converted DEM from meters to feet.")
    elif 0 <= med <= 30:
        print("DEM values already appear to be in feet — no conversion applied.")
    else:
        print(
            f"WARNING: Median sampled value {med:.2f} is outside expected range "
            f"(0–30 ft). Proceeding without unit conversion."
        )

# ---------------------------------------------------------------------------
# 6. Attach elevation column
# ---------------------------------------------------------------------------
gdf["elev_ft_navd88"] = sampled

# ---------------------------------------------------------------------------
# 7. Summary statistics
# ---------------------------------------------------------------------------
valid_mask  = ~gdf["elev_ft_navd88"].isna()
n_valid     = valid_mask.sum()
n_nan       = (~valid_mask).sum()
elev_valid  = gdf.loc[valid_mask, "elev_ft_navd88"]

print(f"\nElevation sampling results:")
print(f"  Valid (not NaN)  : {n_valid:,}")
print(f"  NaN (outside DEM): {n_nan:,}")
if n_valid > 0:
    print(f"  Min    : {elev_valid.min():.3f} ft")
    print(f"  Max    : {elev_valid.max():.3f} ft")
    print(f"  Mean   : {elev_valid.mean():.3f} ft")
    print(f"  Median : {elev_valid.median():.3f} ft")

# ---------------------------------------------------------------------------
# 8. Fill NaN with median of 5 nearest valid-elevation parcels
# ---------------------------------------------------------------------------
if n_nan > 0 and n_valid >= KNN:
    # Build KD-tree in EPSG:6437 (metric) coordinates
    valid_idx = np.where(valid_mask)[0]
    nan_idx   = np.where(~valid_mask)[0]

    # Centroid coords in parcel CRS (metric — EPSG:6437)
    all_cx = centroids.x.values
    all_cy = centroids.y.values

    tree = KDTree(np.column_stack([all_cx[valid_idx], all_cy[valid_idx]]))
    _, nn_positions = tree.query(
        np.column_stack([all_cx[nan_idx], all_cy[nan_idx]]),
        k=min(KNN, n_valid),
    )

    filled = 0
    for i, row_pos in enumerate(nan_idx):
        neighbour_elevs = gdf["elev_ft_navd88"].iloc[valid_idx[nn_positions[i]]]
        fill_val = float(neighbour_elevs.median())
        gdf.at[gdf.index[row_pos], "elev_ft_navd88"] = fill_val
        filled += 1

    print(f"\nNaN parcels filled with nearest-{KNN} median: {filled:,}")
elif n_nan > 0:
    print(f"\nNot enough valid parcels ({n_valid}) to perform KNN fill — NaN retained.")
else:
    print("\nNo NaN elevations to fill.")

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
pathlib.Path("data").mkdir(exist_ok=True)
gdf.to_file(str(PARCELS_OUT), driver="GPKG")
print(f"\nSaved → {PARCELS_OUT}  ({len(gdf):,} features)")

# ---------------------------------------------------------------------------
# 10. Histogram
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)

elev_all = gdf["elev_ft_navd88"].dropna()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(elev_all, bins=20, color="#1565C0", edgecolor="white", linewidth=0.5)

for x_val, label, color in VLINES:
    ax.axvline(x_val, color=color, linestyle="--", linewidth=1.2)
    ax.text(
        x_val + 0.04, ax.get_ylim()[1] * 0.95,
        label, color=color, fontsize=8.5, va="top",
    )

ax.set_xlabel("Elevation (ft, NAVD88)", fontsize=11)
ax.set_ylabel("Number of Parcels", fontsize=11)
ax.set_title(
    "Parcel Elevation Distribution — Brickell / Biscayne Bay Study Area",
    fontsize=12, fontweight="bold",
)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(str(PLOT_OUT), dpi=200)
print(f"Saved plot → {PLOT_OUT}")
