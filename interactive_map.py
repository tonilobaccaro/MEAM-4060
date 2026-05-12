"""
Interactive folium flood risk map for the Brickell study area.

Input  : data/study_parcels_scored_wgs84.gpkg  (EPSG:4326)
Output : outputs/flood_risk_map.html
"""

import json
import pathlib
import sys

import geopandas as gpd
import folium
from folium import GeoJson, GeoJsonPopup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARCELS_IN = pathlib.Path("data/study_parcels_scored_wgs84.gpkg")
MAP_OUT    = pathlib.Path("outputs/flood_risk_map.html")

SAFE_YEAR  = 2076

# ---------------------------------------------------------------------------
# 1. Load parcels (must be EPSG:4326)
# ---------------------------------------------------------------------------
if not PARCELS_IN.exists():
    print(f"ERROR: {PARCELS_IN} not found — run attach_scores.py first.")
    sys.exit(1)

print(f"Loading {PARCELS_IN} …")
gdf = gpd.read_file(str(PARCELS_IN))
assert str(gdf.crs).upper() in {"EPSG:4326", "WGS 84"}, \
    f"Expected EPSG:4326, got {gdf.crs}"
print(f"  {len(gdf):,} parcels  CRS: {gdf.crs}\n")

# ---------------------------------------------------------------------------
# 2. Map center
# ---------------------------------------------------------------------------
cx = float(gdf.geometry.centroid.x.mean())
cy = float(gdf.geometry.centroid.y.mean())

m = folium.Map(
    location=[cy, cx],
    zoom_start=15,
    tiles="CartoDB positron",
)

# ---------------------------------------------------------------------------
# 3. Color function
# ---------------------------------------------------------------------------
def score_to_color(score) -> str:
    """Map a 0-100 risk score to a hex colour."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "#aaaaaa"
    if s <= 20:  return "#1a9641"
    if s <= 40:  return "#a6d96a"
    if s <= 60:  return "#ffffbf"
    if s <= 80:  return "#fdae61"
    return "#d7191c"


def tc_label(val) -> str:
    """Format a T_c year for display."""
    try:
        yr = int(val)
    except (TypeError, ValueError):
        return "N/A"
    return "Safe past 2075" if yr >= SAFE_YEAR else f"Year {yr}"


# ---------------------------------------------------------------------------
# 4. Build GeoJSON dict, annotate each feature with style + popup properties
# ---------------------------------------------------------------------------
POPUP_COLS = [
    "parcel_id", "elev_ft_navd88", "fld_zone",
    "T_c_low", "T_c_intermediate", "T_c_high",
    "risk_score", "risk_category",
]
POPUP_LABELS = {
    "parcel_id":         "Parcel ID",
    "elev_ft_navd88":    "Elevation (ft NAVD88)",
    "fld_zone":          "Flood Zone",
    "T_c_low":           "T_c — Low Scenario",
    "T_c_intermediate":  "T_c — Intermediate Scenario",
    "T_c_high":          "T_c — High Scenario",
    "risk_score":        "Risk Score (0–100)",
    "risk_category":     "Risk Category",
}
avail_pop = [c for c in POPUP_COLS if c in gdf.columns]

TC_COLS = {"T_c_low", "T_c_intermediate", "T_c_high"}


def build_popup_html(props: dict) -> str:
    rows = ""
    for col in avail_pop:
        label = POPUP_LABELS.get(col, col)
        raw   = props.get(col)
        if col in TC_COLS:
            val = tc_label(raw)
        elif col == "risk_score":
            try:
                val = f"{float(raw):.1f}"
            except (TypeError, ValueError):
                val = "N/A"
        else:
            val = "" if raw is None else str(raw)
        rows += (
            "<tr>"
            f"<td style='padding:3px 6px;font-weight:bold;white-space:nowrap'>{label}</td>"
            f"<td style='padding:3px 6px'>{val}</td>"
            "</tr>"
        )
    return (
        "<div style='font-family:sans-serif;font-size:12px;min-width:220px'>"
        "<table style='border-collapse:collapse;width:100%'>"
        f"{rows}"
        "</table></div>"
    )


# Serialise GDF to GeoJSON dict once, then annotate in place
gdf_dict = json.loads(gdf.to_json())

for feat in gdf_dict["features"]:
    props = feat["properties"]
    props["_fill_color"] = score_to_color(props.get("risk_score"))
    props["_popup_html"] = build_popup_html(props)

# Tooltip fields (only those actually present)
tt_fields  = [c for c in ["risk_score", "risk_category", "fld_zone"] if c in gdf.columns]
tt_aliases = {"risk_score": "Risk Score:", "risk_category": "Category:", "fld_zone": "Flood Zone:"}

GeoJson(
    gdf_dict,
    style_function=lambda feat: {
        "fillColor":   feat["properties"].get("_fill_color", "#aaaaaa"),
        "fillOpacity": 0.7,
        "weight":      0.5,
        "color":       "#333333",
    },
    popup=GeoJsonPopup(
        fields=["_popup_html"],
        aliases=[""],
        labels=False,
        parse_html=True,
        max_width=320,
    ),
    tooltip=folium.GeoJsonTooltip(
        fields=tt_fields,
        aliases=[tt_aliases[f] for f in tt_fields],
        style="font-family:sans-serif;font-size:11px;",
    ),
    name="Flood Risk Parcels",
).add_to(m)

# ---------------------------------------------------------------------------
# 5. Custom HTML colorbar legend
# ---------------------------------------------------------------------------
legend_html = """
<div style="
    position: fixed;
    bottom: 40px; right: 15px;
    z-index: 9999;
    background: white;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: sans-serif;
    font-size: 12px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
    min-width: 150px;
">
  <b>Flood Risk Score</b><br>
  <div style="display:flex;align-items:stretch;margin-top:6px">
    <div style="
      width: 18px; flex-shrink: 0; margin-right: 8px; border-radius: 3px;
      border: 1px solid #aaa;
      background: linear-gradient(to top,
        #1a9641 0%, #a6d96a 25%, #ffffbf 50%, #fdae61 75%, #d7191c 100%);
    "></div>
    <div style="display:flex;flex-direction:column;justify-content:space-between;
                line-height:1.2;height:120px">
      <span>100 &mdash; High</span>
      <span>75</span>
      <span>50</span>
      <span>25</span>
      <span>0 &mdash; Low</span>
    </div>
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ---------------------------------------------------------------------------
# 6. Map title
# ---------------------------------------------------------------------------
title_html = """
<div style="
    position: fixed;
    top: 10px; left: 50%; transform: translateX(-50%);
    z-index: 9999;
    background: rgba(255,255,255,0.92);
    border: 1px solid #bbb; border-radius: 6px;
    padding: 8px 18px;
    font-family: sans-serif; font-size: 14px; font-weight: bold;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    pointer-events: none; white-space: nowrap;
">
  Brickell Waterfront &mdash; Flood Risk Score (NOAA 2022 SLR Scenarios)
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

folium.LayerControl(collapsed=False).add_to(m)

# ---------------------------------------------------------------------------
# 7-8. Save
# ---------------------------------------------------------------------------
pathlib.Path("outputs").mkdir(exist_ok=True)
m.save(str(MAP_OUT))
print(f"Map saved to {MAP_OUT} — open in any web browser")
