
import rasterio
import numpy as np
import os
from utils.paths import base_path
from pyproj import Transformer

data_path = os.path.join(base_path, "data", "historic_files")


def get_predictions():
    density_files = []
    for (root, dirs, files) in os.walk(data_path, topdown=True):
        for f in files:
            if "prediction" in f:
                file_path = os.path.join(root, f)
                density_files.append(file_path)
    return density_files


def get_debris_coords(image_path):
    src = rasterio.open(image_path)
    image = src.read(1)
    debris_pixel_coords = np.argwhere(image == 1)
    return debris_pixel_coords


def get_plastic_points(image_path):
    src = rasterio.open(image_path)
    image = src.read(1)
    debris_pixels = np.argwhere(image == 1)
    return debris_pixels


all_point_data = []

for file in get_predictions():
    plastic_pixels = get_plastic_points(file)
    src = rasterio.open(file)
    image = src.read(1)
    meta = src.meta
    geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in plastic_pixels]
    transformer = Transformer.from_crs(src.crs, "epsg:4326")
    coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]
    all_point_data.extend(coords)

import plotly.figure_factory as ff

import pandas as pd
df = pd.DataFrame(all_point_data, columns =['centroid_lat', 'centroid_lon'])
df.to_csv(os.path.join(data_path, "hond"))
if not df.empty:
    fig = ff.create_hexbin_mapbox(
        data_frame=df, lat="centroid_lat", lon="centroid_lon",
        nx_hexagon=15, opacity=0.6, labels={"color": "Point Count"},
        show_original_data=True,
        original_data_marker=dict(size=5, opacity=0.3, color="deeppink"),
        color_continuous_scale="Thermal", min_count=0,

    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=0, l=0, r=0))

    fig.show()