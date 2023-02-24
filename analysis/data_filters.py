import pandas as pd
import os
import sys

import rasterio

from coordinate_generators import generate_plastic_coordinates
from utils.geographic_utils import get_min_max_long_lat

print(sys.path)
sys.path.append("..")
from utils.dir_management import get_files, base_path
from rasterstats import point_query
import numpy as np
from cartopy.geodesic import Geodesic


# Returns the total amount of masking for the image (Clear and Water values considered unmasked).
def get_fmask_percentage(file):
    src = rasterio.open(file)
    image = src.read(1)
    total_pixels = np.count_nonzero(image > 0)
    total_masked_pixels = ((1 < image) & (image < 5)).sum()
    try:
        return (total_masked_pixels / total_pixels) * 100
    except ZeroDivisionError:
        if total_masked_pixels == 0:
            return 0
        else:
            return 100



def mean_pixel_value_filter(df):
    MPM = df.vals.values.tolist()
    q1 = np.quantile(MPM, 0.25)
    # finding the 3rd quartile
    q3 = np.quantile(MPM, 0.75)
    med = np.median(MPM)
    # finding the iqr region
    iqr = q3 - q1
    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    # fig = plt.figure(figsize=(10, 7))
    # # Creating plot
    # plt.boxplot(MPM)
    # # show plot
    # plt.show()
    print(iqr, med, upper_bound, lower_bound, q3)
    for index, row in df.iterrows():
        if row["vals"] > q3:
            df.drop(index=index, inplace=True)
    return df


def calculate_plastic_percent_threshold(plastic_percentages, max_plastic_percentage):
    removed_dates_count = 0
    for percentage in plastic_percentages:
        if percentage >= max_plastic_percentage:
            plastic_percentages.remove(percentage)
            removed_dates_count += 1
    # finding the 1st quartile
    q1 = np.quantile(plastic_percentages, 0.25)
    # finding the 3rd quartile
    q3 = np.quantile(plastic_percentages, 0.75)
    q4 = np.quantile(plastic_percentages, 0.95)
    med = np.median(plastic_percentages)
    # finding the iqr region
    iqr = q3 - q1
    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    print(iqr, med, upper_bound, lower_bound, q3)
    return q4


def bathymetry_filter(df, min_depth, file):
    for index, row in df.iterrows():
        point = {'type': 'Point', 'coordinates': (row["longitude"], row["latitude"])}
        depth = point_query(point, file)[0]
        if depth > min_depth:
            df.drop(index=index, inplace=True)
        else:
            pass
    return df


def class_percentages_filter(data_path, tag, max_plastic_percent, max_masking_percent, land_blurring):
    all_point_data = []
    plastic_percentages = []
    mask_percentages = []
    for file in sorted(get_files(data_path, tag)):
        date = os.path.basename(file).split("_")[1]
        coords = generate_plastic_coordinates(file, date, land_blurring)
        all_point_data.extend(coords[0])
        plastic_percentages.append([coords[1], date])
        mask_percentages.append([coords[2], date])
    df = pd.DataFrame(all_point_data, columns=['latitude', 'longitude', 'date'])
    thresh = calculate_plastic_percent_threshold([item[0] for item in plastic_percentages], max_plastic_percent)
    # remove all dates with excessively high plastic percentage
    for percentage, date in plastic_percentages:
        if percentage > thresh:
            df = df[df.date != date]
    # non optimal duplication of work
    # remove all dates with excessively high cloud masking
    for percentage, date in mask_percentages:
        if percentage > max_masking_percent:
            df = df[df.date != date]
    return df


def create_port_csv():
    arr = np.array([])
    df = pd.read_csv(os.path.join(base_path, "utils", "ports", "WPI.csv"))
    n = df[["PORT_NAME", "LONGITUDE", "LATITUDE"]].to_numpy()
    for i in n:
        arr = np.append(arr, i)
    df2= pd.read_csv(os.path.join(base_path, "utils", "ports", "ports.csv"))
    n2 = df2[["NAME", "X", "Y"]].to_numpy()
    for i in n2:
        arr =np.append(arr, i)
    df3 = pd.read_csv(os.path.join(base_path, "utils", "ports", "anchorage_overrides.csv"))
    n3 = df3[["label", "longitude", "latitude"]].to_numpy()
    for i in n3:
        arr = np.append(arr, i)

    arr = np.reshape(arr, (-1, 3))
    df = pd.DataFrame(arr, columns=['name', 'longitude', 'latitude'])
    df.drop('name')
    df.to_csv("all_ports.csv")


def get_region_ports():
    min_lon, max_lon, min_lat, max_lat = get_min_max_long_lat()
    df = pd.read_csv(os.path.join(base_path, "utils", "ports", "all_ports.csv"))
    for index, row in df.iterrows():
        if min_lon <= float(row["longitude"]) <= max_lon and min_lat <= float(row["latitude"]) <= max_lat:
            pass
        else:
            df.drop(index=index, inplace=True)
    print(df)
    return df


def port_mask(coords_df, distance):
    coords_df = coords_df[['longitude', 'latitude']]
    ports_df = get_region_ports()
    detections = coords_df.to_numpy()
    ports = ports_df[["longitude", "latitude"]].to_numpy()
    source = detections.repeat(len(ports), axis=0)
    dest = np.tile(ports, (len(detections), 1))
    geo = Geodesic()
    distances = geo.inverse(source, dest)[:, 0].reshape((len(detections), len(ports)))
    close = distances < distance
    mask = np.invert(np.any(close, axis=1))
    coords_df.reset_index(drop=True, inplace=True)
    i = 0
    for val in mask:
        if val:
            pass
        else:
            coords_df = coords_df.drop(index=i)
        i += 1
    return coords_df