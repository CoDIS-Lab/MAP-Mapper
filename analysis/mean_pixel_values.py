from math import sin, cos, sqrt, atan2, radians
import numpy as np


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
    trim_level = 0.5  # 50% Trimming
    if L == 0:  # in order to avoid errors in average calculation we directly set hexagons w/o detections to 0.0 MDM
        return 0.0
    else:
        temp = np.sort(x)
        trim = np.round(trim_level * L, 0).astype(int)
        return np.mean(temp[trim:])
