import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from coordinate_generators import generate_plastic_coordinates
print(sys.path)
from utils.dir_management import get_files
from rasterstats import point_query


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
    fig = plt.figure(figsize=(10, 7))
    # Creating plot
    plt.boxplot(MPM)
    # show plot
    plt.show()
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
    #q4 = np.quantile(plastic_percentages, 0.85)
    med = np.median(plastic_percentages)
    # finding the iqr region
    iqr = q3 - q1
    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    print(iqr, med, upper_bound, lower_bound, q3)
    return upper_bound


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


