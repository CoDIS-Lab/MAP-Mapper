# gets all EPSG:4326 coordinates from a prediction mask where plastic has been classified
import numpy as np
import rasterio
from global_land_mask import globe
from pyproj import Transformer


def generate_plastic_coordinates(file, date, land_blurring):
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
        lon = np.arange(point_lon - land_blurring, point_lon + land_blurring, 0.001)
        point_lat = coord[0]
        lat = np.arange(point_lat - land_blurring, point_lat + land_blurring, 0.001)
        if globe.is_land(lat, point_lon).any() or globe.is_land(point_lat, lon).any():
            discarded += 1
        else:
            dated_coord = list(coord)
            dated_coord.append(date)
            dated_coords.append(dated_coord)

    total_pixels = np.count_nonzero(image > 0)
    total_masked_pixels = np.count_nonzero(image == 3)
    mask_percentage = (total_masked_pixels / total_pixels) * 100
    total_water_pixels = np.count_nonzero(image == 2)
    total_plastic_pixels = np.count_nonzero(image == 1) - discarded
    total_unmasked_pixels = total_water_pixels + total_plastic_pixels
    plastic_percentage = (total_plastic_pixels / total_unmasked_pixels) * 100
    print(f"{date} plastic percentage: {plastic_percentage}")
    print(f"{date} mask percentage: {mask_percentage}")
    return dated_coords, plastic_percentage, mask_percentage


def generate_plastic_coordinates2(file, date):
    src = rasterio.open(file)
    meta = src.meta
    image = src.read(1)
    plastic_pixels = np.argwhere(image == 1)
    # transform numpy coordinates to geo-coordinates
    geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in plastic_pixels]
    transformer = Transformer.from_crs(src.crs, "epsg:4326")
    coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]

    total_pixels = np.count_nonzero(image > 0)
    total_masked_pixels = np.count_nonzero(image == 3)
    mask_percentage = (total_masked_pixels / total_pixels) * 100
    total_water_pixels = np.count_nonzero(image == 2)
    total_plastic_pixels = np.count_nonzero(image == 1)
    total_unmasked_pixels = total_water_pixels + total_plastic_pixels
    plastic_percentage = (total_plastic_pixels / total_unmasked_pixels) * 100
    print(f"{date} plastic percentage: {plastic_percentage}")
    print(f"{date} mask percentage: {mask_percentage}")
    dated_coords = []
    for coord in coords:
        dated_coord = list(coord)
        dated_coord.append(date)
        dated_coord.append(plastic_percentage)
        dated_coord.append(mask_percentage)
        dated_coords.append(dated_coord)

    return dated_coords


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

