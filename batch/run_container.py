import shutil
import sys


from fmask_api.f_mask import run_fmask
from masking.prediction_masker import mask_prediction, crop_f_mask, apply_threshold
import os
from utils.dir_management import setup_directories, breakdown_directories, base_path, get_files, clear_downloads

sys.path.insert(0, os.path.join(base_path, "analysis"))
sys.path.insert(0, os.path.join(base_path, "acolite-main"))
sys.path.insert(0, os.path.join(base_path, "acolite_api"))
sys.path.insert(0, os.path.join(base_path, "semantic_segmentation"))
sys.path.insert(0, os.path.join(base_path, "sentinel_downloader"))
sys.path.insert(0, os.path.join(base_path, "smooth_patches"))
from analysis import save_coordinates_to_csv2
from data_filters import get_fmask_percentage
from sentinelsat import read_geojson,SentinelAPI
from image_engineer.image_engineering import ImageEngineer
from utils.geographic_utils import get_crs
import pandas as pd
from semantic_segmentation.debris_predictor import create_image_prediction
from dotenv import load_dotenv
from acolite_api.acolite_processor import run_acolite

load_dotenv()
os.environ['PROJ_LIB'] = '/home/henry/anaconda3/envs/mapper/share/proj'
os.environ['PROJ_DEBUG'] = "3"

# code for command line interface
def run_container(product_id):
    
    downloads_dir = os.path.join(base_path, "data", "unprocessed")
    user_name = os.environ.get('USER_NAME')
    password = os.environ.get('PASSWORD')
    # set fmask threshold (if masking > this %, tile discarded to reduce impact of false positives)
    fmask_threshold = 35
    # set threshold (only pixels that the model predicts as having >99% chance of being plastic are classified as plastic
    threshold = 0.99
    # print to terminal for easier manual verification
    print(f"Running MAP-mapper for {product_id}")
    #set up dirs for processing product
    setup_directories()
    # connect to sentinel hub
    api = SentinelAPI(user_name, password, 'https://apihub.copernicus.eu/apihub')
    # download SAFE file
    api.download(product_id, downloads_dir)
    bundles = os.listdir(downloads_dir)
    print(bundles)
    if bundles:
        for product in bundles:
            # set tile_id and date
            tile_id = product.split("_")[-2]
            date = bundles[0].split("_")[2][:8]

            # run f-mask on each sentinel SAFE file
            run_fmask(os.path.join(base_path, "data", "unprocessed"))

            # verifies the scene does not contain excessive masking
            fmask_files = get_files(os.path.join(base_path, "data", "merged_geotiffs"), "cloud")
            fmask_percentage = None
            for file in fmask_files:
                if file.endswith("cloud.tif"):
                    fmask_percentage = get_fmask_percentage(file)
            #only process scenes with less than 35% masking

            if fmask_percentage < fmask_threshold:
                print(f"Low cloud detected ({fmask_percentage}%). Processing scene...")
                run_acolite(product)
                print("processing files........")
                image_engineer = ImageEngineer(id=tile_id, date=date, land_mask=True, cloud_mask=True)

                # get crs of SAFE file
                image_engineer.crs = get_crs()

                # load processed acolite rhos images (assigns file path to self.tiff_files)
                image_engineer.load_images()

                # combines processed satellite images output by acolite processor (one for each band)
                image_engineer.combine_bands()

                geotiff = os.path.join(base_path, "data", "unmerged_geotiffs", tile_id + "_" + date + ".tif")

                # patch full ROI for predictions
                image_engineer.patch_image(geotiff)

                # make predictions on image patches
                create_image_prediction()

                # merge predicted masks into one file
                image_engineer.merge_tiles(directory=os.path.join(base_path, "data", "predicted_patches"),
                                            mode="probs")

                # merge f-masks into one large mask
                image_engineer.merge_tiles(directory=os.path.join(base_path, "data", "merged_geotiffs"),
                                            mode="clouds")

                # apply threshold
                apply_threshold(os.path.join(base_path, "data", "merged_geotiffs"), threshold)

                # read coords for f-mask crop
                poly = read_geojson(os.path.join(base_path, "poly.geojson"))

                # crop f-mask for ROI
                crop_f_mask(tile_id, date, poly, image_engineer.crs)

                # apply f-mask to predictions, generate and apply land-mask
                mask_prediction(id=image_engineer.id, date=image_engineer.date,
                                land_mask=image_engineer.land_mask, cloud_mask=image_engineer.cloud_mask)

                # get plastic coordinates and save to csv
                save_coordinates_to_csv2(os.path.join(base_path, "data", "merged_geotiffs"), "prediction_masked")

                # plot single date coordinates - Currently broken, plot data by specifying the output data_path in analysis.py
                # plot_data_single_day(date)
            else:
                print(f"Excessive cloud detected ({fmask_percentage}%). Skipping scene...")
            # clean data dirs for next iteration, save predictions and tif to output files dir
            breakdown_directories(date)
            if os.path.exists(os.path.join(base_path, "data", "processed")):
                shutil.rmtree(os.path.join(base_path, "data", "processed"))
        clear_downloads()
