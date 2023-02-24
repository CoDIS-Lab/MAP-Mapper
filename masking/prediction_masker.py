import numpy as np
import shapely
import rasterio.mask
import shapefile
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
import geojson
import os
from utils.dir_management import get_files, base_path
from utils.geographic_utils import project_wsg_shape_to_crs, transform_raster


# this crops f-mask output mask to the ROI
def crop_f_mask(id, date, polygon, crs):
    projected_shape = project_wsg_shape_to_crs(shapely.geometry.shape(polygon), crs)
    print(crs)
    with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs",
                                    id + "_" + date + "_cloud.tif")) as dataset:
        out_image, out_transform = rasterio.mask.mask(dataset, [projected_shape], crop=True)
        out_meta = dataset.meta.copy()
        # cloud-mask is 20m resolution, but needs to be same size in pixels as sat image
        out_meta.update(
            {"transform": out_transform, "height": out_image.shape[1] * 2, "width": out_image.shape[2] * 2})
        with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs",
                                        id + "_" + date + "_cloud_cropped.tif"), "w", **out_meta) as src:
            src.write(out_image)


# This function creates a clipped shp file from the worldwide shp file downloaded to the utils directory
# The shp file is available from https://osmdata.openstreetmap.de/data/land-polygons.html, (large polygons not split)
# This new shp files is saved as land_mask.shp in the utils dir and is clipped using the coordinates provided by poly.geojson file
def create_land_mask():
    with open(os.path.join(base_path, "poly.geojson")) as f:
        gj = geojson.load(f)
    features = gj['coordinates'][0]

    # x min and x max coords
    xmin = min([coord[0] for coord in features])
    xmax = max([coord[0] for coord in features])
    ymin = min([coord[1] for coord in features])
    ymax = max([coord[1] for coord in features])
    # only non-convex 4 sided polygons
    clipped_shp = gpd.read_file(os.path.join(base_path, "utils", "world_land", "land_polygons.shp"), bbox=Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]))
    # Save clipped shapefile
    clipped_shp.to_file(os.path.join(base_path, "utils", "region_land", "land_mask.shp"), driver="ESRI Shapefile")


# apply the land mask to the file (prediction mask)
def apply_land_mask(file_path):
    # get crs of original file
    with rasterio.open(file_path) as src:
        crs = src.crs
    # raster must be same CRS system as shp file
    # transform predictions mask to EPSG:4326 (geojson crs) so that the land mask can be applied
    transform_raster(file_path, 'EPSG:4326')
    with rasterio.open(file_path) as src:
        # get land mask polygons
        sf = shapefile.Reader(os.path.join(base_path, "utils", "region_land", "land_mask.shp"))
        shapes = sf.shapes()
        # apply mask, invert is used to mask land, rather than ocean
        out_image, out_transform = rasterio.mask.mask(src, shapes, invert=True)
        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        # overwrites old file with new masked file

        with rasterio.open(file_path, "w", **out_meta) as dest:
            dest.write(out_image)
    # transform file back to original CRS
    with rasterio.open(file_path, "w", **out_meta) as dest:
        dest.write(out_image)
    transform_raster(file_path, str(crs))


def apply_cloud_mask(prediction, masked_prediction_file, flag_file, out_meta):
    print("applying cloud mask...")
    with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", flag_file)) as mask_ds:
        # Only consider mask values == 5 (these are tagged as water by fmask).
        mask = mask_ds.read(1) != 5
        mask = mask[:prediction.shape[0], :prediction.shape[1]]
        masked_unet = np.copy(prediction)
        # Convert all non-water pixels to category 3 (clouds).
        masked_unet[mask] = 3
        with rasterio.open(masked_prediction_file, "w", **out_meta) as masked_prediction:
            masked_prediction.write(np.expand_dims(masked_unet, axis=0))


def mask_prediction(id, date, land_mask=True, cloud_mask=True):
    predict_file = [x for x in os.listdir(os.path.join(base_path, "data", "merged_geotiffs")) if x.endswith("_prediction.tif")][0]
    prob_file = [x for x in os.listdir(os.path.join(base_path, "data", "merged_geotiffs")) if x.endswith("_probabilities.tif")][0]
    files_to_mask = [predict_file, prob_file]
    for file in files_to_mask:
        # either prediction or probabilities
        suffix = file.split("_")[-1].strip(".tif")
        with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", file)) as model_output:
            img = model_output.read(1)
            out_meta = model_output.meta.copy()
            flag_file = [x for x in os.listdir(os.path.join(base_path, "data", "merged_geotiffs")) if "_cloud_cropped" in x][0]
            masked_prediction_file = os.path.join(base_path, "data", "merged_geotiffs", id + "_" + date + "_" + suffix + "_masked.tif")
            if cloud_mask:
                apply_cloud_mask(img, masked_prediction_file, flag_file, out_meta)
            else:
                with rasterio.open(masked_prediction_file, "w", **out_meta) as masked_prediction:
                    masked_prediction.write(np.expand_dims(img, axis=0))
        # checks for land mask, creates it if it doesn't exist
        # land_mask must be deleted manually if investigating new region of interest
        if land_mask:
            if not os.path.exists(os.path.join(base_path, "utils", "region_land", "land_mask.shp")):
                print("no land mask found, creating...")
                create_land_mask()
            # apply land mask to prediction file
            print("applying land mask...")
            apply_land_mask(masked_prediction_file)


# applys thresholding to model output probabilities GeoTiff to produce binary prediction mask.
def apply_threshold(dir, threshold):
    prob_files = get_files(dir, "probabilities")
    for file in prob_files:
        src = rasterio.open(file)
        meta = src.meta
        image = src.read(1)
        # only pixels above or equal to threshold are classified as plastic
        image[(image >= threshold) & (image < 1)] = 1
        # make all probs below threshold into water
        image[image < threshold] = 2
        # write threshold prediction
        with rasterio.open(file.strip("probabilities.tif") + "prediction.tif", "w", **meta) as threshold_prediction:
            threshold_prediction.write(np.expand_dims(image, axis=0))


# this is not used in the main pipeline. It is only to be used if masking other output files to mask predictions or probabilities.
# This is required for visualising results for multiple different thresholds (run as main).
def mask_many_predictions(dir, tag,  land_mask, cloud_mask, suffix):
    files_to_mask = []
    for (root, dirs, files) in os.walk(dir, topdown=True):
        for f in files:
            if f.endswith(tag+ ".tif"):
                file_path = os.path.join(root, f)
                files_to_mask.append(file_path)
    for file in files_to_mask:
        name = file.split("/")[-1].strip(tag + ".tif")
        current_dir = file.split("/")[-2]

        with rasterio.open(os.path.join(dir, file)) as predictions:
            prediction = predictions.read(1)
            out_meta = predictions.meta.copy()

            masked_prediction_file = os.path.join(dir, current_dir, name + suffix + ".tif")

            if cloud_mask:
                print("applying cloud mask on " + name)
                flag_file = [x for x in os.listdir(os.path.join(dir, current_dir)) if "cloud_cropped" in x][0]
                with rasterio.open(os.path.join(dir, current_dir, flag_file)) as mask_ds:
                    # Only consider mask values == 5 (these are tagged as water by fmask).
                    mask = mask_ds.read(1) != 5
                    mask = mask[:prediction.shape[0], :prediction.shape[1]]
                    masked_unet = np.copy(prediction)
                    # Convert all non-water pixels to category 3 (clouds).
                    masked_unet[mask] = 3
                    with rasterio.open(masked_prediction_file, "w", **out_meta) as masked_prediction:
                        masked_prediction.write(np.expand_dims(masked_unet, axis=0))
            if land_mask:
                # checks for land mask, creates it if it doesn't exist
                # land_mask must be deleted manually if investigating new region of interest
                if not os.path.exists(os.path.join(base_path, "utils", "region_land", "land_mask.shp")):
                    print("no land mask found, creating...")
                    create_land_mask()
                # apply land mask to prediction file
                print("applying land mask on " + name)
                if os.path.exists(masked_prediction_file):
                    apply_land_mask(masked_prediction_file)
                else:
                    prediction_file = os.path.join(dir, current_dir, name + "_probabilities.tif")
                    apply_land_mask(prediction_file)

if __name__ == "__main__":
    print("Running manual masking")
    # directory of files for masking
    data_path = os.path.join(base_path, "data", "outputs")
    data_path = "/home/henry/Desktop/dissertation_data/cornwall/historic_files"
    # apply threshold to probability file
    #apply_threshold(data_path, 0.99)
    mask_many_predictions(data_path, "probabilities", land_mask=True, cloud_mask=True, suffix="probabilities_masked")
    # mask_prediction("T51PTR", "20220507", True, True)