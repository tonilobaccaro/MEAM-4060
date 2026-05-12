"""
Load Miami-Dade County parcel data, filter to the Brickell / Biscayne Bay
study area, optionally filter to residential land use, and save/plot results.
"""

import pathlib
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
SHP_PATH   = pathlib.Path("data/MDC_Parcels.shp")
OUT_GPKG   = pathlib.Path("data/study_parcels.gpkg")
OUT_PLOT   = pathlib.Path("outputs/study_area_parcels.png")

TARGET_CRS = "EPSG:6437"
PLOT_CRS   = "EPSG:3857"   # contextily requires Web Mercator

# Brickell / Biscayne Bay study area in WGS84
BBOX_WGS84 = {"west": -80.200, "east": -80.185, "south": 25.755, "north": 25.775}

# Miami-Dade DOR land-use codes to keep (residential / condo)
RESIDENTIAL_CODES = {"01", "04", "08", "11"}

# Candidate land-use column names, in priority order
LU_CANDIDATES = ["DOR_UC", "LAND_USE", "LU_CODE"]

# Candidate parcel-ID column names, in priority order
ID_CANDIDATES = ["FOLIO", "FOLIO_1", "PARCEL_ID", "APN"]

# ---------------------------------------------------------------------------
# 1. Load shapefile
# ---------------------------------------------------------------------------
if not SHP_PATH.exists():
    print(
        "ERROR: Parcel shapefile not found.\n"
        "  Download MDC Parcels shapefile from\n"
        "  https://gis-mdc.opendata.arcgis.com\n"
        f"  and place it at {SHP_PATH}"
    )
    sys.exit(1)

print(f"Loading {SHP_PATH} …")
gdf = gpd.read_file(str(SHP_PATH))

print(f"\nTotal parcels loaded : {len(gdf):,}")
print(f"Columns ({len(gdf.columns)}):")
for col in gdf.columns:
    print(f"  {col}")

# ---------------------------------------------------------------------------
# 2. Reproject to EPSG:6437
# ---------------------------------------------------------------------------
print(f"\nReprojecting to {TARGET_CRS} …")
gdf = gdf.to_crs(TARGET_CRS)

# ---------------------------------------------------------------------------
# 3. Bounding-box filter — convert bbox corners to EPSG:6437 first
# ---------------------------------------------------------------------------
transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)

xmin, ymin = transformer.transform(BBOX_WGS84["west"], BBOX_WGS84["south"])
xmax, ymax = transformer.transform(BBOX_WGS84["east"], BBOX_WGS84["north"])

print(
    f"\nStudy-area bbox in {TARGET_CRS}:\n"
    f"  xmin={xmin:,.1f}  xmax={xmax:,.1f}\n"
    f"  ymin={ymin:,.1f}  ymax={ymax:,.1f}"
)

gdf_study = gdf.cx[xmin:xmax, ymin:ymax].copy()

# ---------------------------------------------------------------------------
# 4. Report study-area count
# ---------------------------------------------------------------------------
print(f"\nParcels in study area : {len(gdf_study):,}")

# ---------------------------------------------------------------------------
# 5. Land-use filter
# ---------------------------------------------------------------------------
lu_col = next((c for c in LU_CANDIDATES if c in gdf_study.columns), None)

if lu_col:
    print(f"\nLand-use column found : '{lu_col}'")
    print("Value counts (all study-area parcels):")
    print(gdf_study[lu_col].astype(str).value_counts().to_string())

    # Normalise both sides: strip whitespace and leading zeros before comparing
    # so "01", "1", and 1 (int) all match code "01".
    res_stripped = {c.lstrip("0") or "0" for c in RESIDENTIAL_CODES}
    lu_norm = gdf_study[lu_col].astype(str).str.strip().str.lstrip("0").replace("", "0")
    keep_mask = lu_norm.isin(res_stripped)

    gdf_study = gdf_study[keep_mask].copy()
    print(
        f"\nAfter residential filter (codes {sorted(RESIDENTIAL_CODES)}) : "
        f"{len(gdf_study):,} parcels"
    )
else:
    print(
        "\nWARNING: No land-use column found "
        f"(tried {LU_CANDIDATES}). Skipping residential filter."
    )

# ---------------------------------------------------------------------------
# 6. parcel_id column
# ---------------------------------------------------------------------------
id_col = next((c for c in ID_CANDIDATES if c in gdf_study.columns), None)

if id_col:
    gdf_study["parcel_id"] = gdf_study[id_col].astype(str)
    print(f"\nparcel_id sourced from column '{id_col}'.")
else:
    gdf_study["parcel_id"] = "parcel_" + gdf_study.index.astype(str)
    print("\nNo FOLIO/parcel-ID column found; parcel_id assigned from row index.")

# ---------------------------------------------------------------------------
# 7. Save GeoPackage
# ---------------------------------------------------------------------------
pathlib.Path("data").mkdir(exist_ok=True)
gdf_study.to_file(str(OUT_GPKG), driver="GPKG")
print(f"\nSaved → {OUT_GPKG}  ({len(gdf_study):,} features)")

# ---------------------------------------------------------------------------
# 8. Plot — with optional contextily basemap
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)

try:
    import contextily as ctx
    has_ctx = True
except ImportError:
    has_ctx = False
    print("contextily not available — plotting without basemap.")

fig, ax = plt.subplots(figsize=(10, 12))

if has_ctx:
    gdf_plot = gdf_study.to_crs(PLOT_CRS)
else:
    gdf_plot = gdf_study

gdf_plot.plot(
    ax=ax,
    facecolor="none",
    edgecolor="#E53935",
    linewidth=0.6,
)

if has_ctx:
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=15)
    ax.set_xlabel("Easting — Web Mercator (m)", fontsize=9)
    ax.set_ylabel("Northing — Web Mercator (m)", fontsize=9)
else:
    ax.set_xlabel(f"Easting ({TARGET_CRS}, m)", fontsize=9)
    ax.set_ylabel(f"Northing ({TARGET_CRS}, m)", fontsize=9)

ax.set_title(
    "Study Area Parcels — Brickell / Biscayne Bay, Miami FL",
    fontsize=12, fontweight="bold",
)
ax.tick_params(axis="both", labelsize=7)

fig.tight_layout()
fig.savefig(str(OUT_PLOT), dpi=200)
print(f"Saved plot → {OUT_PLOT}")
