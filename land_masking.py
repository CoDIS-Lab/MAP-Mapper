import rasterio.mask
import shapefile

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Polygon
import geojson
from utils.paths import base_path
import os

data_path = os.path.join(base_path, "data", "historic_files")


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


def create_land_mask():
    with open(os.path.join(base_path, "poly.geojson")) as f:
        gj = geojson.load(f)
    features = gj['coordinates'][0]
    shp = gpd.read_file(os.path.join(base_path, "utils", "land_polygons.shp"))

    xmin, xmax, ymin, ymax = features[0][0], features[1][0], features[0][1], features[1][1]
    # Selection box
    selection_box = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
    clipped_shp = gpd.clip(shp, selection_box)

    # Save clipped shapefile
    clipped_shp.to_file(os.path.join(base_path, "utils", "land_mask.shp"), driver="ESRI Shapefile")


def apply_land_mask(file_path):
    # transform predictions mask to EPSG:4326 (geojson crs) so that the land mask can be applied
    with rasterio.open(file_path) as src:
        crs = src.crs
    transform_raster(file_path, 'EPSG:4326')
    with rasterio.open(file_path) as src:
            sf = shapefile.Reader(os.path.join(base_path, "utils", "land_mask.shp"))
            shapes = sf.shapes()
            out_image, out_transform = rasterio.mask.mask(src, shapes, invert=True)
            out_meta = src.meta

            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            with rasterio.open(file_path, "w", **out_meta) as dest:
                dest.write(out_image)
    transform_raster(file_path, str(crs))
