#!/usr/bin/env bash
set -euo pipefail

echo "Installing Python dependencies for geospatial flood risk analysis..."

pip install --upgrade \
    geopandas \
    fiona \
    shapely \
    pyproj \
    rasterio \
    matplotlib \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    requests \
    folium \
    mapclassify \
    contextily \
    jupyter \
    tqdm

echo ""
echo "--- Installed versions ---"
python -c "import geopandas; print('geopandas:', geopandas.__version__)"
python -c "import rasterio; print('rasterio: ', rasterio.__version__)"
python -c "import sklearn;   print('scikit-learn:', sklearn.__version__)"

echo ""
echo "--- Checking GDAL ---"
if command -v gdal-config &> /dev/null; then
    echo "GDAL version: $(gdal-config --version)"
else
    echo "WARNING: gdal-config not found. GDAL does not appear to be installed." \
         "Some geospatial operations may fail. Install GDAL via your system package" \
         "manager (e.g. 'sudo apt install gdal-bin libgdal-dev' on Debian/Ubuntu)."
fi
