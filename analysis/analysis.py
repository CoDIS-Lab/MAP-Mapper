import rasterio
import numpy as np
import os
from utils.dir_management import get_files, base_path
from pyproj import Transformer
import plotly.figure_factory as ff
import pandas as pd

# plotly_params
hex_bin_number = 10  # higher number results in smaller and more numerous hex_bins
hex_bin_opacity = 0.5  # 0 (invisible) to 1 (opaque)
data_marker_size = 3  # plotted plastic detection marker size
data_marker_opacity = 0.5  # 0 (invisible) to 1 (opaque)
min_detection_count = 0  # lowest number of plastic detections needed for a hex_bin to be visualised on the map


# gets all EPSG:4326 coordinates from a prediction mask where plastic has been classified
def generate_plastic_coordinates(file, date):
    src = rasterio.open(file)
    meta = src.meta
    image = src.read(1)
    plastic_pixels = np.argwhere(image == 1)
    # transform numpy coordinates to geo-coordinates
    geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in plastic_pixels]
    transformer = Transformer.from_crs(src.crs, "epsg:4326")
    coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]
    dated_coords = []
    for coord in coords:
        dated_coord = list(coord)
        dated_coord.append(date)
        dated_coords.append(dated_coord)
    return dated_coords


# code to find and plot all suspected plastic on a map
def plot_data(tag, data_path):
    all_point_data = []
    for file in sorted(get_files(data_path, tag)):
        date = os.path.basename(file).split("_")[1]
        all_point_data.extend(generate_plastic_coordinates(file, date))

    df = pd.DataFrame(all_point_data, columns =['centroid_lat', 'centroid_lon', 'date'])
    if not df.empty:
        fig = ff.create_hexbin_mapbox(
            data_frame=df, lat="centroid_lat", lon="centroid_lon",
            nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
            show_original_data=True,
            original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
            color_continuous_scale="Reds", min_count=min_detection_count,

        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=0, l=0, r=0))
        fig.show()


# This code was used to threshold probabilities and plot the new predictions.
# This removes the need to generate new prediction masks for each threshold before plotting.
def generate_threshold_coords(file, threshold):
    src = rasterio.open(file)
    meta = src.meta
    image = src.read(1)
    # ignore all
    image[image > 1] = np.nan
    plastic_pixels = np.argwhere(image > threshold)
    # transform numpy coordinates to geo-coordinates
    geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in
                  plastic_pixels]
    transformer = Transformer.from_crs(src.crs, "epsg:4326")
    coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]
    return coords


# use to threshold probabilities file
def plot_probabilities(tag, data_path, threshold):
    all_point_data = []
    for file in sorted(get_files(data_path, tag)):
        all_point_data.extend(generate_threshold_coords(file, threshold))

    df = pd.DataFrame(all_point_data, columns =['centroid_lat', 'centroid_lon'])
    if not df.empty:
        fig = ff.create_hexbin_mapbox(
            data_frame=df, lat="centroid_lat", lon="centroid_lon",
            nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
            show_original_data=True,
            original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
            color_continuous_scale="Reds", min_count=min_detection_count,

        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=0, l=0, r=0))
        fig.show()


def save_coordinates_to_csv(data_path, tag):
    all_point_data = []
    for file in sorted(get_files(data_path, tag)):
        date = os.path.basename(file).split("_")[1]
        all_point_data.extend(generate_plastic_coordinates(file, date))
    df = pd.DataFrame(all_point_data, columns=['centroid_lat', 'centroid_lon', 'date'])
    if os.path.exists(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv")):
        df.to_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv"), mode="a", header=False)
    else:
        df.to_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv"), mode="w", header=True)


def plot_data_single_day(date):
    df = pd.read_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv"))
    if not df.empty:
        is_date = df['date'] == int(date)
        df = df[is_date]
        fig = ff.create_hexbin_mapbox(
            title="Plastic Detections for " + date,
            data_frame=df, lat="centroid_lat", lon="centroid_lon",
            nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
            show_original_data=True,
            original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
            color_continuous_scale="Reds", min_count=min_detection_count,
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=40, l=0, r=0))
        fig.show()
    else:
        print("no detections for " + date + " skipping plot...")


if __name__ == "__main__":
    print("Analysing MAP-Mapper outputs and plotting plastic detections...")
    data_path = os.path.join(base_path, "data", "outputs")
    plot_data("prediction_masked", data_path)
    plot_probabilities("probabilities_masked", data_path, 0.815)