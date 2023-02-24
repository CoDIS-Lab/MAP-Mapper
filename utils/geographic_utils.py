import os
from xml.etree import ElementTree as ET

import geojson
import pyproj
import shapely
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import rasterio
from utils.dir_management import base_path


# returns crs of sentinel-2 region
def get_crs():
    xml_path = None
    # data path of sentinel SAFE file
    data_path = os.path.join(base_path, "data", "unprocessed")
    for (root, dir, files) in os.walk(data_path):
        for f in files:
            # file containing coordinate reference system information
            if f == "MTD_TL.xml":
                xml_path = os.path.join(root, f)
                break
        if xml_path:
            break
    # get data from xml document
    # get data from xml document
    tree = ET.parse(xml_path)
    root = tree.getroot()
    epsg = root.findall('.//HORIZONTAL_CS_CODE')[0].text
    return str(epsg)


# This function converts geojson polygon format to a the correct crs to enable fmask to be transposed onto the geotiff
def project_wsg_shape_to_crs(shape, crs):
    project = lambda x, y: pyproj.transform(
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(init=crs),
        x,
        y
    )
    return shapely.ops.transform(project, shape)


# This function transforms a raster from one CRS system to another
def transform_raster(file_path, dst_crs):
    with rasterio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(file_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def get_min_max_long_lat():
    with open(os.path.join(base_path, "poly.geojson")) as f:
        gj = geojson.load(f)
    features = gj['coordinates'][0]
    longs = [x[0] for x in features]
    lats = [x[1] for x in features]
    min_lon = min(longs)
    max_lon = max(longs)
    min_lat = min(lats)
    max_lat = max(lats)
    return min_lon, max_lon, min_lat, max_lat
