import sys
import urllib.request
import urllib.error
import os
from statistics import mean
from utils.dir_management import base_path
import geojson
import json


def check_wind(max_wind_speed, date):
    if max_wind_speed:
        with open(os.path.join(base_path, "poly.geojson")) as f:
            gj = geojson.load(f)
        features = gj['coordinates'][0]

        # x min and x max coords
        xmin = min([coord[0] for coord in features])
        xmax = max([coord[0] for coord in features])
        ymin = min([coord[1] for coord in features])
        ymax = max([coord[1] for coord in features])
        # get mid-point of polygon for location reference
        location = str(mean([float(ymax), float(ymin)])) + "," + str(mean([float(xmax), float(xmin)]))
        # build API QUERY
        BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
        ApiKey = os.environ.get('WEATHER')
        Include = "days"
        Elements = "windspeed,windgust,conditions"
        # basic query including location
        ApiQuery = BaseURL + location
        # append date
        ApiQuery += "/" + date[:4] + "-" + date[4:6] + "-" + date[6:8]
        # Url is completed. Now add query parameters (could be passed as GET or POST)
        ApiQuery += "?"
        # append each parameter as necessary - limit query to daily data and wind and conditions
        ApiQuery += "&include=" + Include
        ApiQuery += "&include=" + Elements
        ApiQuery += "&key=" + ApiKey

        try:
            response = urllib.request.urlopen(ApiQuery)
            string = response.read().decode('utf-8')
            weather_conditions = json.loads(string)
        except urllib.error.HTTPError as e:
            ErrorInfo = e.read().decode()
            print('Error code: ', e.code, ErrorInfo)
            sys.exit()
        except urllib.error.URLError as e:
            ErrorInfo = e.read().decode()
            print('Error code: ', e.code, ErrorInfo)
            sys.exit()
        wind_speed = float(weather_conditions["days"][0]["windspeed"])
        if wind_speed < float(max_wind_speed[0]):
            return True
        else:
            return False
    else:
        return True