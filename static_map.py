"""
Publication-quality static flood risk map for the Brickell study area.

Input  : data/study_parcels_scored.gpkg  (EPSG:6437)
Output : outputs/risk_map_static.png     (300 DPI)
"""

import pathlib
import sys

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PARCELS_IN  = pathlib.Path("data/study_parcels_scored.gpkg")
PLOT_OUT    = pathlib.Path("outputs/risk_map_static.png")

SRC_CRS     = "EPSG:6437"
PLOT_CRS    = "EPSG:3857"    # Web Mercator for contextily
CMAP        = "RdYlGn_r"     # reversed: red = high risk, green = low
VMIN, VMAX  = 0, 100
FIG_SIZE    = (10, 8)
DPI         = 300
SCALE_M     = 200            # scale bar length in metres

# ---------------------------------------------------------------------------
# 1. Load parcels
# ---------------------------------------------------------------------------
if not PARCELS_IN.exists():
    print(f"ERROR: {PARCELS_IN} not found — run attach_scores.py first.")
    sys.exit(1)

print(f"Loading {PARCELS_IN} …")
gdf = gpd.read_file(str(PARCELS_IN)).to_crs(SRC_CRS)
print(f"  {len(gdf):,} parcels  CRS: {gdf.crs}")

# ---------------------------------------------------------------------------
# 2. Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

# ---------------------------------------------------------------------------
# 3. Reproject to EPSG:3857
# ---------------------------------------------------------------------------
gdf_plot = gdf.to_crs(PLOT_CRS)

# ---------------------------------------------------------------------------
# 4. Plot parcels with continuous RdYlGn_r colormap
# ---------------------------------------------------------------------------
cmap = mplcm.get_cmap(CMAP)
norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

# geopandas .plot() with column + cmap handles NaN parcels as gray by default
gdf_plot.plot(
    ax=ax,
    column="risk_score",
    cmap=CMAP,
    vmin=VMIN,
    vmax=VMAX,
    edgecolor="#444444",
    linewidth=0.3,
    missing_kwds={"color": "#cccccc", "edgecolor": "#888888", "label": "No data"},
    legend=False,    # we'll add a custom colorbar below
)

# ---------------------------------------------------------------------------
# 5. Contextily basemap
# ---------------------------------------------------------------------------
try:
    import contextily as ctx
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        zoom=15,
        crs=PLOT_CRS,
    )
    print("  Basemap added via contextily.")
except ImportError:
    print("WARNING: contextily not available — basemap skipped.")
except Exception as e:
    print(f"WARNING: Basemap fetch failed ({e}) — skipping.")

# ---------------------------------------------------------------------------
# 6. Colorbar on the right
# ---------------------------------------------------------------------------
divider = make_axes_locatable(ax)
cax     = divider.append_axes("right", size="4%", pad=0.08)

cb = ColorbarBase(
    cax,
    cmap=cmap,
    norm=norm,
    orientation="vertical",
    ticks=[0, 25, 50, 75, 100],
)
cb.set_label("Flood Risk Score (0–100)", fontsize=10, labelpad=8)
cb.ax.set_yticklabels(["0\n(Low)", "25", "50", "75", "100\n(High)"], fontsize=8)

# ---------------------------------------------------------------------------
# 7. North arrow
# ---------------------------------------------------------------------------
ax.annotate(
    "↑N",
    xy=(0.96, 0.94),
    xycoords="axes fraction",
    ha="center", va="center",
    fontsize=14, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555555", alpha=0.8),
)

# ---------------------------------------------------------------------------
# 8. Scale bar (200 m in EPSG:3857 — scale factor ≈ 1/cos(lat))
# ---------------------------------------------------------------------------
# Compute the pixel-equivalent of SCALE_M metres at the map's latitude.
# In EPSG:3857 x-units are metres multiplied by the Mercator distortion
# factor 1/cos(lat), so we need to correct back to true metres.
import math
lat_rad  = math.radians(25.765)           # study area latitude
merc_m   = SCALE_M / math.cos(lat_rad)   # metres in EPSG:3857 east-direction

# Place the scale bar in the lower-left corner (axes fraction 0.05-0.05)
xmin_ax, xmax_ax = ax.get_xlim()
ymin_ax, ymax_ax = ax.get_ylim()
ax_width_m  = xmax_ax - xmin_ax
ax_height_m = ymax_ax - ymin_ax

sb_x0 = xmin_ax + 0.04 * ax_width_m
sb_y  = ymin_ax + 0.05 * ax_height_m
sb_x1 = sb_x0 + merc_m

ax.plot([sb_x0, sb_x1], [sb_y, sb_y],
        color="black", linewidth=2.5, solid_capstyle="butt", zorder=6)
# End ticks
tick_h = 0.005 * ax_height_m
for sx in (sb_x0, sb_x1):
    ax.plot([sx, sx], [sb_y - tick_h, sb_y + tick_h],
            color="black", linewidth=2.0, zorder=6)

ax.text(
    (sb_x0 + sb_x1) / 2, sb_y + 1.8 * tick_h,
    f"{SCALE_M} m",
    ha="center", va="bottom", fontsize=8,
    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1),
    zorder=7,
)

# ---------------------------------------------------------------------------
# 9. Title
# ---------------------------------------------------------------------------
ax.set_title(
    "Brickell Waterfront Parcel Flood Risk\n"
    "NOAA Intermediate SLR Scenario (2025–2075)",
    fontsize=12, fontweight="bold", pad=10,
)

# ---------------------------------------------------------------------------
# 10. Remove axis tick labels
# ---------------------------------------------------------------------------
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(left=False, bottom=False)

# ---------------------------------------------------------------------------
# 11. Save
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)
fig.savefig(str(PLOT_OUT), dpi=DPI, bbox_inches="tight")
print(f"\nSaved → {PLOT_OUT}  ({DPI} DPI)")
