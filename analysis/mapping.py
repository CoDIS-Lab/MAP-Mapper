import rasterio
import numpy as np
import os
from os import path
import logging
import sys
print(sys.path)
import utils
from utils.dir_management import get_files, base_path
from pyproj import Transformer
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import sin, cos, sqrt, atan2, radians
import sys
import plotly.io as pio

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

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def get_coordinates(tag, data_path, fname):
    '''
    calculating mdm values for each pixels and returning detected MD&SP pixels in a csv file with their corresponding MDM values
    '''
    k = 0
    for file in sorted(get_files(data_path, tag)):
        src = rasterio.open(file)

        meta = src.meta
        image = src.read(1)
        if k == 0:
            I = np.empty(image.shape)
            I2 = np.empty(image.shape)
        image[image == 99] = 0
        image[image == 3] = 0
        image[np.isnan(image)] = 0
        I += image # probability maps after removing land masked values
        image[image>0.99] = 1
        image[image != 1] = 0
        I2 += image # thresholded detection maps 
        k += 1  
        print("File {} is processed!".format(k))  
    I = I/k # taking average of probability maps
    I3 = I*(100*I2/k) # MDM calculation in pixel level
    detect_pixels = np.argwhere(I3 > 0.01) # getting rid of higly small MDM values
    pix = len(detect_pixels)

    geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in detect_pixels]
    transformer = Transformer.from_crs(src.crs, "epsg:4326")
    coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]
    vals = []
    for c, i in enumerate(detect_pixels):
        vals.append(I3[i[0], i[1]])
        print("Processed {}/{}".format(c,pix))
    all_point_data2 = []
    plastic_percentages = []
    for i in range(len(coords)):
        dated_coord = list(coords[i])
        dated_coord.append(vals[i])
        all_point_data2.append(dated_coord)
    
    df3 = pd.DataFrame(all_point_data2, columns =['centroid_lat', 'centroid_lon', 'vals'])
    fname3 = fname+"4.csv"
    df3.to_csv(fname3, index=False)



    for file in sorted(get_files(data_path, tag)):
        date = os.path.basename(file).split("_")[1]
        coords = generate_plastic_coordinates(file, date)
        all_point_data.extend(coords[0])
        plastic_percentages.append([coords[1], date])
    df = pd.DataFrame(all_point_data, columns =['centroid_lat', 'centroid_lon', 'date'])
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
    trim_level = 0.5     # 50% Trimming
    if L == 0:           # in order to avoid errors in average calculation we directly set hexagons w/o detections to 0.0 MDM 
        return 0.0
    else:
        temp = np.sort(x)
        trim = np.round(trim_level*L, 0).astype(int)
        return np.mean(temp[trim:])

def plot_maps(file):
    df3 = pd.read_csv(file+"4.csv")
    c1  = [np.abs(df3.centroid_lat.mean()), np.abs(df3.centroid_lon.max())]
    c2 = [np.abs(df3.centroid_lat.mean()), np.abs(df3.centroid_lon.min())]
    hexagon_size = np.round(latlon2distance(c1, c2)/5).astype(int)
    df4 = df3.nlargest(10, 'vals') # taking 10 largest MDM values
    if not df3.empty:
        px.set_mapbox_access_token("pk.eyJ1Ijoib2thcmFrdXMiLCJhIjoiY2w5bjlmd28xMDRrbzN2czVubzJ6eWFueSJ9.MWPE4mLxoJjv3Cmr1N6OQA")
        fig = ff.create_hexbin_mapbox(
            data_frame=df3, lat="centroid_lat", lon="centroid_lon", color='vals', agg_func=my_mean,
            nx_hexagon=hexagon_size, opacity=0.5, labels={"color": "MPM"},
            original_data_marker=dict(size=3, opacity=0.0, color="crimson"),
            show_original_data=True, range_color=(0,1.5), # setting max to 1.5. this is manual but for better visualisation
            color_continuous_scale="Greys", min_count=1)
        fig.add_trace(go.Scattermapbox( # this is for adding maximum MDM values
            lat=df4.centroid_lat,
            lon=df4.centroid_lon,
            mode='markers', showlegend=False,
            marker=go.scattermapbox.Marker(size=8, color=df4.vals, colorbar=dict(orientation='h', yanchor="bottom",
            y=-0.15, xanchor="center", title = 'MDM', x=0.5, len=0.9), opacity=0.99, cmin=0, cmax=17, colorscale='Rainbow')))
        fig.update_layout(mapbox_style="streets", width=700, height=600)
        fig.update_layout(coloraxis_colorbar=dict(title="50%<br> Trimmed<br> Mean<br> MDM"))
        fig.show()


if __name__ == "__main__":
    ##################################################
    # to run the code for Manila in command line you need to type
    # python mapping3.py "Manila"
    #################################################
    file = "outputs_" + sys.argv[1]
    data_path = os.path.join(base_path, file)
    if path.exists(file+"4.csv"): # if you run the code once, it will create the csv file and after that everytime you run this code it plots those calculated results. If you change something and need a new plot, please delete the csv file and run afterwrds.
        plot_maps(file)
    else:
        get_coordinates("probabilities_masked", data_path, file) # we are taking land masked probability maps directly.
        plot_maps(file)
    
