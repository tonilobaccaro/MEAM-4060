"""
Load FEMA National Flood Hazard Layer (NFHL) data for Miami-Dade County.
Supports either a file geodatabase (NFHL_12086C.gdb) or individual shapefiles.
Reprojects to EPSG:6437, summarises, saves GeoPackages, and plots an overview.
"""

import pathlib
import sys

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GDB_PATH     = pathlib.Path("data/NFHL_12086C.gdb")
SHP_HAZ_PATH = pathlib.Path("data/S_FLD_HAZ_AR.shp")
SHP_BFE_PATH = pathlib.Path("data/S_BFE.shp")

TARGET_CRS        = "EPSG:6437"
LAYER_HAZ         = "S_FLD_HAZ_AR"
LAYER_BFE         = "S_BFE"
OUT_HAZ_GPKG      = pathlib.Path("data/fema_flood_zones.gpkg")
OUT_BFE_GPKG      = pathlib.Path("data/fema_bfe_lines.gpkg")
OUT_PLOT          = pathlib.Path("outputs/fema_zones_overview.png")

ZONE_COLORS = {
    "AE": "#2196F3",   # blue
    "VE": "#F44336",   # red
    "X":  "#4CAF50",   # green
}
DEFAULT_COLOR = "#9E9E9E"   # gray for all other zones

# ---------------------------------------------------------------------------
# 1. Locate data source
# ---------------------------------------------------------------------------
use_gdb = GDB_PATH.exists()
use_shp = (not use_gdb) and SHP_HAZ_PATH.exists()

if not use_gdb and not use_shp:
    print(
        "ERROR: No FEMA data found.\n"
        "  Option A (geodatabase) : download the FEMA NFHL for Miami-Dade from\n"
        "    https://msc.fema.gov and place it at data/NFHL_12086C.gdb\n"
        "  Option B (shapefiles)  : place S_FLD_HAZ_AR.shp and S_BFE.shp in data/"
    )
    sys.exit(1)

if use_gdb:
    print(f"Source: geodatabase  → {GDB_PATH}\n")
    layers = fiona.listlayers(str(GDB_PATH))
    print(f"Layers in {GDB_PATH.name} ({len(layers)} total):")
    for name in layers:
        print(f"  {name}")
    print()
else:
    print(f"Source: shapefiles   → {SHP_HAZ_PATH.parent}\n")
    print("(Geodatabase not found; falling back to individual shapefiles.)\n")


def read_layer(layer_name: str) -> gpd.GeoDataFrame:
    """Read a layer from the GDB or its equivalent shapefile."""
    if use_gdb:
        return gpd.read_file(str(GDB_PATH), layer=layer_name)
    shp = pathlib.Path(f"data/{layer_name}.shp")
    if not shp.exists():
        print(f"ERROR: {shp} not found.")
        sys.exit(1)
    return gpd.read_file(str(shp))


# ---------------------------------------------------------------------------
# 2. Flood hazard areas (S_FLD_HAZ_AR)
# ---------------------------------------------------------------------------
print("=" * 60)
print("S_FLD_HAZ_AR — Flood Hazard Areas")
print("=" * 60)

haz = read_layer(LAYER_HAZ)
haz = haz.to_crs(TARGET_CRS)

print(f"CRS             : {haz.crs}")
print(f"Total features  : {len(haz):,}")
print(f"\nColumns ({len(haz.columns)}):")
for col in haz.columns:
    print(f"  {col}")

if "FLD_ZONE" in haz.columns:
    print("\nFLD_ZONE value counts:")
    print(haz["FLD_ZONE"].value_counts().to_string())
else:
    print("\nWARNING: FLD_ZONE column not found; available columns listed above.")

print("\nFirst 5 rows:")
print(haz.head(5).to_string())
print()

# ---------------------------------------------------------------------------
# 3. Base flood elevation lines (S_BFE)
# ---------------------------------------------------------------------------
print("=" * 60)
print("S_BFE — Base Flood Elevation Lines")
print("=" * 60)

bfe = read_layer(LAYER_BFE)
bfe = bfe.to_crs(TARGET_CRS)

print(f"Total BFE lines : {len(bfe):,}")
print(f"\nColumns ({len(bfe.columns)}):")
for col in bfe.columns:
    print(f"  {col}")

# BFE_LEN holds the actual elevation value in the NFHL schema
bfe_col = "BFE_LEN" if "BFE_LEN" in bfe.columns else None
if bfe_col:
    vals = bfe[bfe_col].dropna()
    print(f"\n{bfe_col} statistics:")
    print(f"  min  : {vals.min():.3f}")
    print(f"  max  : {vals.max():.3f}")
    print(f"  mean : {vals.mean():.3f}")
else:
    print("\nWARNING: BFE_LEN column not found; skipping elevation statistics.")
print()

# ---------------------------------------------------------------------------
# 4. Save GeoPackages
# ---------------------------------------------------------------------------
pathlib.Path("data").mkdir(exist_ok=True)

haz.to_file(str(OUT_HAZ_GPKG), driver="GPKG")
print(f"Saved → {OUT_HAZ_GPKG}")

bfe.to_file(str(OUT_BFE_GPKG), driver="GPKG")
print(f"Saved → {OUT_BFE_GPKG}\n")

# ---------------------------------------------------------------------------
# 5. Overview plot
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 14))

if "FLD_ZONE" in haz.columns:
    # Assign a plot color to every row up front (avoids per-category loops)
    haz["_color"] = haz["FLD_ZONE"].map(
        lambda z: ZONE_COLORS.get(str(z).strip().upper(), DEFAULT_COLOR)
    )

    # Plot each color group in one call for efficiency
    for color, group in haz.groupby("_color"):
        group.plot(ax=ax, color=color, edgecolor="none", linewidth=0)

    # Legend
    legend_entries = [
        mpatches.Patch(color=c, label=f"Zone {z}")
        for z, c in ZONE_COLORS.items()
    ] + [mpatches.Patch(color=DEFAULT_COLOR, label="Other")]
    ax.legend(handles=legend_entries, fontsize=10, loc="lower right")
else:
    haz.plot(ax=ax, color=DEFAULT_COLOR, edgecolor="none")

bfe.plot(ax=ax, color="black", linewidth=0.4, alpha=0.6, label="BFE lines")

ax.set_title(
    "FEMA Flood Hazard Zones — Miami-Dade County (EPSG:6437)",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Easting (m)", fontsize=10)
ax.set_ylabel("Northing (m)", fontsize=10)
ax.tick_params(axis="both", labelsize=8)

fig.tight_layout()
fig.savefig(str(OUT_PLOT), dpi=150)
print(f"Saved plot → {OUT_PLOT}")
