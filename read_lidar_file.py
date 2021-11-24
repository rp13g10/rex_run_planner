#%%
import rasterio as rio
import shapefile as shp
import xml

#%%
with rio.open('data/LIDAR-DTM-1m-2020-SU42se/SU42se_DTM_1m.tif') as fobj:
    lidar = fobj.read(1)

#%%
sf = shp.Reader('data/LIDAR-DTM-1m-2020-SU42se/index/SU42se_DTM_1M.shp')
print(sf.bbox)
# %%
from convertbng.util import convert_lonlat, convert_bng
print(
    convert_lonlat(sf.bbox[slice(0, 4, 2)], sf.bbox[slice(1, 4, 2)])
)

#%%
def get_elevation(lat, lon):
    # Convert to grid reference
    tgt = convert_bng(lon, lat)
    easting = tgt[0][0]
    northing = tgt[1][0]

    # Get top-left boundary
    boundary_e = sf.bbox[0]
    boundary_n = sf.bbox[3]

    # Calculate required offset from this point
    offset_e = easting - boundary_e
    offset_s = boundary_n - northing

    # Return elevation at this offset
    elevation = lidar[int(offset_s), int(offset_e)]

    return elevation

#%%
# Fishers Pond Restaurant
get_elevation(50.9865759649451, -1.305346815483719)

# Retirement village, top of bishopstoke hill
get_elevation(50.976724087240385, -1.3369273449909098)
# NOTE
# Propose keeping elevation in the current BNG format
# Use convertbng.util.convert_bng to go from long/lat to grid
# Get point elevation from grid via array indexing
# %%
