"""
Environment verification for the geospatial flood risk analysis project.
Imports key packages, prints their versions, and runs a minimal CRS sanity check.
"""

import sys

PACKAGES = [
    ("geopandas", "geopandas"),
    ("rasterio",  "rasterio"),
    ("folium",    "folium"),
    ("sklearn",   "scikit-learn"),
    ("contextily","contextily"),
    ("scipy",     "scipy"),
]

modules = {}
failed = False

for module_name, pip_name in PACKAGES:
    try:
        mod = __import__(module_name)
        modules[module_name] = mod
        version = getattr(mod, "__version__", "unknown")
        print(f"  {pip_name:<14} {version}")
    except ImportError as exc:
        print(f"  ERROR: could not import '{pip_name}' ({module_name}): {exc}")
        failed = True

if failed:
    print("\nOne or more packages failed to import. Run setup.sh to install missing dependencies.")
    sys.exit(1)

print("\n--- Sanity check: GeoDataFrame CRS reprojection ---")

import shapely.geometry
geopandas = modules["geopandas"]

point = shapely.geometry.Point(-80.19, 25.76)
gdf = geopandas.GeoDataFrame(
    {"name": ["Brickell, Miami"]},
    geometry=[point],
    crs="EPSG:4326",
)

gdf_projected = gdf.to_crs("EPSG:6437")
x, y = gdf_projected.geometry.iloc[0].x, gdf_projected.geometry.iloc[0].y

print(f"  Input  (EPSG:4326)  lon={point.x}, lat={point.y}")
print(f"  Output (EPSG:6437)  x={x:.3f} m, y={y:.3f} m")
print("\nAll checks passed.")
