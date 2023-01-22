import sys
import geojson
import numpy as np
import shapely
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
from dotenv import load_dotenv
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from utils.dir_management import base_path
sys.path.insert(0, os.path.join(base_path, "analysis"))
from sentinel_loader import SentinelLoader
from utils.geographic_utils import get_min_max_long_lat
from global_land_mask import globe
load_dotenv()
from shapely.geometry import Point, Polygon, MultiPoint
import shapely.wkt
user_name = os.environ.get('USER_NAME')
password = os.environ.get('PASSWORD')
weather_key = os.environ.get('WEATHER')
SL = SentinelLoader(start_date='20220101', end_date='20220202', max_cloud_percentage='20', max_wind_speed='0')
SL.get_product_data()
print(SL)

if SL.products:
    min_lon, max_lon, min_lat, max_lat = get_min_max_long_lat()
    longitudes = np.arange(min_lon, max_lon, 0.1)

    latitudes = np.arange(min_lat, max_lat, 0.1)
    ocean_coords = []
    for lat in latitudes:
        for lon in longitudes:
            if globe.is_ocean(lat, lon):
                # lon lat format for polygon intersection
                ocean_coords.append([lon, lat])

    SL_products = SL.products.copy()
    for key, val in SL.products.items():
        valid_points = []
        polygon = shapely.wkt.loads(val['footprint'])
        points = MultiPoint(ocean_coords)
        p = polygon.intersection(points)
        if p.is_empty:
            del SL_products[key]

    SL.products = SL_products
    print(SL.products.items())
