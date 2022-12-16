import matplotlib.pyplot as plt
from global_land_mask import globe
import rasterio
import numpy as np
import os
from os import path
import sys
print(sys.path)
from utils.dir_management import get_files, base_path
from pyproj import Transformer
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import sin, cos, sqrt, atan2, radians
import sys
from rasterio import merge
# plotly_params
hex_bin_number = 6  # higher number results in smaller and more numerous hex_bins
hex_bin_opacity = 0.5  # 0 (invisible) to 1 (opaque)
data_marker_size = 2  # plotted plastic detection marker size
data_marker_opacity = 0.4  # 0 (invisible) to 1 (opaque)
min_detection_count = 0  # lowest number of plastic detections needed for a hex_bin to be visualised on the map
max_plastic_percentage = 0.02 # remove all dates
close_to_land_mask = 1 # mask detections close to land (in km)
plastic_percentages = []
land_blur = 0.008999*close_to_land_mask

def excessive_masking_filter(data_path, limit):
    excessive_dates = []
    for file in sorted(get_files(data_path, "cloud_cropped")):
        src = rasterio.open(file)
        image = src.read(1)
        date = os.path.basename(file).split("_")[1]
        total_pixels = np.count_nonzero(image > 0)
        cloud_shadow = np.count_nonzero(image == 3)
        cloud = np.count_nonzero(image == 2)
        mask_percentage = (float(cloud+cloud_shadow)/ float(total_pixels)) * 100
        if mask_percentage > limit:
            excessive_dates.append(date)
    return excessive_dates
def calculate_plastic_percent_threshold():
    removed_dates_count = 0
    for percentage in plastic_percentages:
        if percentage >= max_plastic_percentage:
            plastic_percentages.remove(percentage)
            removed_dates_count += 1
    # # shift upperbound if dates removed
    # mean_drift = statistics.mean(plastic_percentages)
    # for i in range(0, len(plastic_percentages)//2):
    #     plastic_percentages.append(mean_drift)
    # finding the 1st quartile
    q1 = np.quantile(plastic_percentages, 0.25)
    # finding the 3rd quartile
    q3 = np.quantile(plastic_percentages, 0.75)
    med = np.median(plastic_percentages)
    # finding the iqr region
    iqr = q3 - q1
    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    print(iqr, med, upper_bound, lower_bound, q3)
    # fig = plt.figure(figsize=(10, 7))
    # # Creating plot
    # plt.boxplot(plastic_percentages)
    # # show plot
    # plt.show()
    return upper_bound

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
    discarded = 0
    for coord in coords:
        point_lon = coord[1]
        lon = np.arange(point_lon - land_blur, point_lon + land_blur, 0.001)
        point_lat = coord[0]
        lat = np.arange(point_lat - land_blur, point_lat + land_blur, 0.001)
        if globe.is_land(lat, point_lon).any() or globe.is_land(point_lat, lon).any():
            discarded += 1
        else:
            dated_coord = list(coord)
            dated_coord.append(date)
            dated_coords.append(dated_coord)
    total_water_pixels = np.count_nonzero(image == 3)
    print("water_pixels: " + str(total_water_pixels))
    total_plastic_pixels = np.count_nonzero(image == 1) - discarded
    print("plastic_pixels: " + str(total_plastic_pixels))
    total_unmasked_pixels = total_water_pixels + total_plastic_pixels
    plastic_percentage = (float(total_plastic_pixels) / float(total_unmasked_pixels)) * 100
    plastic_percentages.append(plastic_percentage)
    print(f"{date} plastic percentage: {plastic_percentage}")
    return dated_coords, plastic_percentage


def get_data(data_path, tag):
    all_point_data = []
    plastic_percentages = []
    for file in sorted(get_files(data_path, tag)):
        date = os.path.basename(file).split("_")[1]
        coords = generate_plastic_coordinates(file, date)
        all_point_data.extend(coords[0])
        plastic_percentages.append([coords[1], date])
    df = pd.DataFrame(all_point_data, columns=['centroid_lat', 'centroid_lon', 'date'])
    thresh = calculate_plastic_percent_threshold()
    for percentage, date in plastic_percentages:
        if percentage >= thresh:
            df = df[df['date'] != date]

    # non optimal duplication of work
    for date in excessive_masking_filter(data_path, 40):
        try:
            df = df[df['date'] != date]
        except KeyError:
            print("date already removed due to excessive plastic percentage")
    return df
# code to find and plot all suspected plastic on a map
def plot_data(tag, data_path):
    df = get_data(data_path, tag)
    # google map heatmap csv
    locations = df.copy()
    locations.rename(columns={'centroid_lat': 'latitude', 'centroid_lon': 'longitude'}, inplace=True)
    locations.drop(columns="date", axis=1, inplace=True)
    locations.to_csv(os.path.join(base_path, "data", "outputs", "BOH_filtered_gmaps_heatmap.csv"), index=False)
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
            color_continuous_scale="Reds", min_count=min_detection_count

        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=0, l=0, r=0))
        fig.show()


def save_coordinates_to_csv(data_path, tag):
    all_point_data = []
    for file in sorted(get_files(data_path, tag)):
        date = os.path.basename(file).split("_")[1]
        all_point_data.extend(generate_plastic_coordinates(file, date))
    df = pd.DataFrame(all_point_data, columns=['centroid_lat', 'centroid_lon', 'date'])
    if os.path.exists(os.path.join(data_path, "plastic_coordinates.csv")):
        df.to_csv(os.path.join(data_path, "plastic_coordinates.csv"), mode="a", header=False)
    else:
        df.to_csv(os.path.join(data_path, "plastic_coordinates.csv"), mode="w", header=True)


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


def latlon2distance(c1, c2):
    '''
    latitude/longitude to distance transformation code
    '''
    R = 6373.0
    lat1 = radians(c1[0])
    lon1 = radians(c1[1])
    lat2 = radians(c2[0])
    lon2 = radians(c2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# def get_coordinates(tag, data_path, fname):
#     '''
#     calculating mdm values for each pixels and returning detected MD&SP pixels in a csv file with their corresponding MDM values
#     '''
#     k = 0
#     for file in sorted(get_files(data_path, tag)):
#         src = rasterio.open(file)
#         meta = src.meta
#         image = src.read(1)
#         if k == 0:
#             I = np.empty(image.shape)
#             I2 = np.empty(image.shape)
#         image[image == 99] = 0
#         image[image == 3] = 0
#         image[np.isnan(image)] = 0
#         try:
#             I += image  # probability maps after removing land masked values
#         except:pass
#         image[image > 0.99] = 1
#         image[image != 1] = 0
#         try:
#             I2 += image  # thresholded detection maps
#         except:pass
#         k += 1
#         print("File {} is processed!".format(k))
#
#     I = I / k  # taking average of probability maps
#     I3 = I * (100 * I2 / k)  # MDM calculation in pixel level
#     detect_pixels = np.argwhere(I3 > 0.01)  # getting rid of higly small MDM values
#     pix = len(detect_pixels)
#
#     geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in
#                   detect_pixels]
#     transformer = Transformer.from_crs(src.crs, "epsg:4326")
#     coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]
#     vals = []
#     for c, i in enumerate(detect_pixels):
#         vals.append(I3[i[0], i[1]])
#         print("Processed {}/{}".format(c, pix))
#     all_point_data2 = []
#     for i in range(len(coords)):
#         dated_coord = list(coords[i])
#         dated_coord.append(vals[i])
#         all_point_data2.append(dated_coord)
#
#     df3 = pd.DataFrame(all_point_data2, columns=['centroid_lat', 'centroid_lon', 'vals'])
#     fname3 = fname + "4.csv"
#     df3.to_csv(fname3, index=False)


def get_coordinates(tag, data_path, fname):
    '''
    calculating mdm values for each pixels and returning detected MD&SP pixels in a csv file with their corresponding MDM values
    '''
    k = 0
    df = get_data(data_path, "prediction_masked")
    dates = pd.unique(df["date"])
    temp_files = []
    for file in sorted(get_files(data_path, tag)):
        date = os.path.basename(file).split("_")[1]
        if date not in dates:
            pass
        else:
            src = rasterio.open(file)
            meta = src.meta
            image = src.read(1)
            image[image == 99] = 0
            image[image > 1] = 0
            image[np.isnan(image)] = 0
            pure_probability = file.strip("probabilities_masked.tif") + "probability_layer.tif"
            with rasterio.open(pure_probability, "w", **meta) as dst:
                dst.write(image,indexes=1)
            temp_files.append(pure_probability)
            image[image > 0.99] = 1
            image[image != 1] = 0
            pure_detections = file.strip("probabilities_masked.tif") + "prediction_layer.tif"
            with rasterio.open(pure_detections, "w", **meta) as dst:
                dst.write(image, indexes=1)
            temp_files.append(pure_detections)
    probability_files_to_mosaic = []
    for file in sorted(get_files(data_path, "probability_layer")):
        src = rasterio.open(file)
        probability_files_to_mosaic.append(src)
    prob_mosaic, out_trans = merge.merge(probability_files_to_mosaic, method='sum', nodata=0)
    predictions_files_to_mosaic = []
    for file in sorted(get_files(data_path, "prediction_layer")):
        src = rasterio.open(file)
        predictions_files_to_mosaic.append(src)
    pred_mosaic, out_trans = merge.merge(predictions_files_to_mosaic, method='sum', nodata=0)
    for file in temp_files:
        if file.endswith("prediction_layer.tif") or file.endswith("probability_layer.tif"):
            os.remove(file)
    I = prob_mosaic[0] / (len(temp_files)/2)  # taking average of probability maps
    I3 = I * (100 * pred_mosaic[0] / (len(temp_files)/2))  # MDM calculation in pixel level
    detect_pixels = np.argwhere(I3 > 0.01)  # getting rid of higly small MDM values
    pix = len(detect_pixels)

    geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in
                  detect_pixels]
    transformer = Transformer.from_crs(src.crs, "epsg:4326")
    coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]
    vals = []
    for c, i in enumerate(detect_pixels):
        vals.append(I3[i[0], i[1]])
        print("Processed {}/{}".format(c, pix))
    all_point_data2 = []
    for i in range(len(coords)):
        dated_coord = list(coords[i])
        dated_coord.append(vals[i])
        all_point_data2.append(dated_coord)

    df3 = pd.DataFrame(all_point_data2, columns=['centroid_lat', 'centroid_lon', 'vals'])
    # fname3 = fname + "4.csv"
    # df3.to_csv(fname3, index=False)
    df = df.drop(['date'], axis=1)
    df4 = df.merge(df3, on=["centroid_lat", "centroid_lon"])
    df4 = df4.dropna()
    df4 = df4.drop_duplicates()
    fname3 = fname + "4.csv"
    df4.to_csv(fname3, index=False)


def my_max(x):
    '''
    my own max function to use in hexagon maps
    '''
    if len(x) == 0:
        return 0.0
    else:
        return np.max(x)


def my_mean(x):
    '''
    my own trimmed mean function to use in hexagon maps
    '''
    L = len(x)
    trim_level = 0.05  # 50% Trimming
    if L == 0:  # in order to avoid errors in average calculation we directly set hexagons w/o detections to 0.0 MDM
        return 0.0
    else:
        temp = np.sort(x)
        trim = np.round(trim_level * L, 0).astype(int)
        return np.mean(temp[trim:])


def plot_maps(file):
    df3 = pd.read_csv(file + "4.csv")
    c1 = [np.abs(df3.centroid_lat.mean()), np.abs(df3.centroid_lon.max())]
    c2 = [np.abs(df3.centroid_lat.mean()), np.abs(df3.centroid_lon.min())]
    hexagon_size = np.round(latlon2distance(c1, c2) / 5).astype(int)
    df4 = df3.nlargest(10, 'vals')  # taking 10 largest MDM values
    if not df3.empty:
        px.set_mapbox_access_token(
            "pk.eyJ1Ijoib2thcmFrdXMiLCJhIjoiY2w5bjlmd28xMDRrbzN2czVubzJ6eWFueSJ9.MWPE4mLxoJjv3Cmr1N6OQA")
        fig = ff.create_hexbin_mapbox(
            data_frame=df3, lat="centroid_lat", lon="centroid_lon", color='vals', agg_func=my_mean,
            nx_hexagon=hexagon_size, opacity=0.5, labels={"color": "MPM"},
            original_data_marker=dict(size=3, opacity=0.0, color="crimson"),
            show_original_data=True, range_color=(0, 4),
            # setting max to 1.5. this is manual but for better visualisation
            color_continuous_scale="Greys", min_count=1)
        fig.add_trace(go.Scattermapbox(  # this is for adding maximum MDM values
            lat=df4.centroid_lat,
            lon=df4.centroid_lon,
            mode='markers', showlegend=False,
            marker=go.scattermapbox.Marker(size=8, color=df4.vals, colorbar=dict(orientation='h', yanchor="bottom",
                                                                                 y=-0.15, xanchor="center", title='MDM',
                                                                                 x=0.5, len=0.9), opacity=0.99, cmin=0,
                                           cmax=17, colorscale='Rainbow')))
        fig.update_layout(mapbox_style="streets", width=700, height=600)
        fig.update_layout(coloraxis_colorbar=dict(title="50%<br> Trimmed<br> Mean<br> MDM"))
        fig.show()




if __name__ == "__main__":

   #  print("Analysing MAP-Mapper outputs and plotting plastic detections...")
   #  #data_path = os.path.join(base_path, "data", "outputs")
   # # data_path ="/home/henry/Downloads/argentina"
   #  #data_path = "/home/henry/Desktop/dissertation_data/greece/historic_files"
   #  #data_path = "/home/henry/Desktop/dissertation_data/cornwall/historic_files"
   #  data_path = "/home/henry/Desktop/dissertation_data/BOH-2/historic_files"
   #  #save_coordinates_to_csv(data_path, "thresh_masked99")
   #  plot_data("thresh_masked99", data_path)
   #  #plot_data("prediction", data_path)
   #  #plot_data("prediction_masked", data_path)
   #  # plot_probabilities("probabilities_masked", data_path, 0.95)

    ##################################################
    # to run the code for Manila in command line you need to type
    # python mapping3.py "Manila"
    #################################################
    file = "outputs"
    file = "historic_files"
   # data_path = os.path.join(base_path, "data", file)
    data_path = "/home/henry/Desktop/dissertation_data/BOH-2"
    if path.exists(
            file + "4.csv"):  # if you run the code once, it will create the csv file and after that everytime you run this code it plots those calculated results. If you change something and need a new plot, please delete the csv file and run afterwrds.
        plot_maps(file)
    else:
        get_coordinates("probabilities_masked", data_path,
                        file)  # we are taking land masked probability maps directly.
        plot_maps(file)