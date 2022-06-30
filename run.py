import sys
sys.path.insert(0, "/acolite_api")
sys.path.insert(0, "/semantic_segmentation")
sys.path.insert(0, "/sentinel_downloader")
sys.path.insert(0, "/smooth_patches")
sys.path.insert(0, "/acolite-main/acolite")
import argparse
import os
import subprocess
from sentinelsat import read_geojson
from semantic_segmentation.dataset_loader import DatasetLoader
from utils.dir_management import setup_directories, clean_directories
from datetime import datetime, timedelta
import pandas as pd
from semantic_segmentation.debris_predictor import predict_with_smooth_blending
from sentinel_downloader.sentinel_loader import SentinelLoader
from paths import base_path
from multiprocessing import Pool
from dotenv import load_dotenv

from acolite_api.acolite_processor import acolite_loader

load_dotenv()
input("press Enter to continue")

if __name__ == "__main__":
    today = datetime.today().strftime("%Y%m%d")
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y%m%d")

    parser = argparse.ArgumentParser(description='A sentinel-2 plastic detection pipeline using the MARIDA dataset')

    subparsers = parser.add_subparsers(help='possible uses', dest='command')
    pipeline = subparsers.add_parser('full', help='run full pipeline for a given ROI')
    # full pipeline arguments
    pipeline.add_argument(
        '-start_date',
        nargs=1,
        default=today,
        type=str,
        help='start_date for sentinel 2 full_pipeline predictions to start (YYYYmmdd)',
        dest='start_date'

    )
    pipeline.add_argument(
        '-end_date',
        nargs=1,
        default=tomorrow,
        type=str,
        help='end_date for sentinel 2 full_pipeline predictions to end (YYYYmmdd)',
        dest="end_date"
    )
    pipeline.add_argument(
        '-cloud_percentage',
        nargs=1,
        default=50,
        type=int,
        help='maximum cloud percentage',
        dest="cloud_percentage"
    )

    # partial pipeline components
    download = subparsers.add_parser(
        'download',
        help='download sentinel-2 data'
    )
    download.add_argument(
        '-date',
        nargs=1,
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help='use with --download to specify the date for data download', dest='date'
    )
    download.add_argument(
        '-cloud_percentage',
        nargs=1,
        type=str,
        default=50,
        dest='cloud_percentage',
        help='use with --download to specify the date for data download'
    )
    acolite = subparsers.add_parser(
        "acolite",
        help='complete acolite processing on SAFE files'
        )
    acolite.add_argument(
        '-date',
        nargs=1,
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help='date of SAFE files for processing',
        dest="date"
        )
    acolite.add_argument(
        '-tile',
        nargs=1,
        type=str,
        help='Tile ID',
        dest="id"
    )

    combine_acolite = subparsers.add_parser(
        'merge_processed',
        help='download sentinel-2 data'
    )
    combine_acolite.add_argument(
        '-date',
        nargs=1,
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help='complete acolite processing on SAFE files',
        dest="date"
    )

    predict = subparsers.add_parser(
        'predict',
        help='make predictions on pre-existing geotiff'
    )
    fmask = subparsers.add_parser(
        'fmask',
        help='mask predictions using fmask for more robust cloud and land detection'
    )

    fmask.add_argument(
        '-date',
        nargs=1,
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help='date of SAFE files for processing',
        dest="date"
        )
    fmask.add_argument(
        '-tile',
        nargs=1,
        type=str,
        help='Tile ID',
        dest="tile"
    )

    clean = subparsers.add_parser(
        'clean',
        help='WARNING! Removes all data associated with sentinel downloads, '
             'processing and predictions in the "data" directory tree'
    )
    args = parser.parse_args()
    options = vars(args)
    print(options)

    if args.command == "full":
        setup_directories()
        start = datetime.strptime(args.start_date[0], "%Y%m%d")
        end = datetime.strptime(args.end_date[0], "%Y%m%d")
        date_generated = pd.date_range(start, end)
        dates = []
        for date in date_generated.strftime("%Y%m%d"):
            dates.append(str(date).replace("_", ""))
        print("Finding SAFE files for " + str(dates))

        for i in range(len(dates)):
            start_date = dates[i]
            end_date = (datetime.strptime(start_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            user_name = os.environ.get('USER_NAME')
            password = os.environ.get('PASSWORD')
            SentinelLoader(start_date=start_date, end_date=end_date, max_cloud_percentage=args.cloud_percentage).run()

            bundles = os.listdir(os.path.join(base_path, "data", "unprocessed"))
            print(bundles)
            # if any data to download and process
            if bundles:
                if __name__ == '__main__':
                    with Pool(10) as p:
                        print(p.map(acolite_loader, bundles))
                print("processing files........")
                DL = DatasetLoader()
                DL.run_pipeline()

    if args.command == "download":
        date = args.date[0]
        print("downloading data for given date: " + date)
        # get day following for range of 1 day
        user_name = os.environ.get('USER_NAME')
        password = os.environ.get('PASSWORD')
        end_date = (datetime.strptime(date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        SentinelLoader(start_date=date, end_date=end_date).run()

    if args.command == "acolite":
        bundles = os.listdir(os.path.join(base_path, "data", "unprocessed"))
        print(bundles)
        # if any data to download and process
        if bundles:
            if __name__ == '__main__':
                print("processing SAFE files with acolite........")
                with Pool(10) as p:
                    print(p.map(acolite_loader, bundles))

    if args.command == "predict_from_acolite":
        DL = DatasetLoader(date=args.date[0], id=args.id[0], crs=args.crs[0])
        # load processed acolite rhos images (assigns file path to self.tiff_files)
        DL.load_images()
        # combines processed satellite images output by acolite processor (one for each band)
        DL.combine_bands()
        # merges sentinel 2 tiles into one large image covering whole region of interest
        # please note, this could be made more efficient by patching each tile, then merging the tiles.
        # However, care must be taken not to lose pixels due to cropping.
        # if using 1 or 2 sentinel tiles, it does not make much difference
        DL.merge_tiles(directory=os.path.join(base_path, "data", "unmerged_geotiffs"), mode="images")
        predict_with_smooth_blending()
        # merge predicted masks into one file
        DL.merge_tiles(directory=os.path.join(base_path, "data", "predicted_unet"), mode="masks")
        script = os.path.join(base_path, "acolite_api", "fmask_script.py")
        subprocess.call([sys.executable, script])
        DL.merge_tiles(directory=os.path.join(base_path, "data", "non_water_mask"), mode="clouds")
        poly = read_geojson(os.path.join(base_path, "poly.geojson"))
        DL.crop_non_water_mask(poly, "epsg:32616")
        DL.mask_predictions()

    if args.command == "combine_acolite":
        DL = DatasetLoader(date=args.date[0], id=args.id[0])
        # load processed acolite rhos images (assigns file path to self.tiff_files)
        DL.load_images()
        # combines processed satellite images output by acolite processor (one for each band)
        DL.combine_bands()
        # merges sentinel 2 tiles into one large image covering whole region of interest
        # please note, this could be made more efficient by patching each tile, then merging the tiles.
        # However, care must be taken not to lose pixels due to cropping.
        # if using 1 or 2 sentinel tiles, it does not make much difference
        DL.merge_tiles(directory=os.path.join(base_path, "data", "unmerged_geotiffs"), mode="images")

    if args.command == "predict":
        DL = DatasetLoader(date=args.date[0], id=args.id[0], crs=args.crs[0])
        predict_with_smooth_blending()
        # merge predicted masks into one file
        DL.merge_tiles(directory=os.path.join(base_path, "data", "predicted_unet"), mode="masks")

    if args.command == "fmask":
        DL = DatasetLoader(date=args.date[0], id=args.tile[0])
        script = os.path.join(base_path, "acolite_api", "fmask_script.py")
        run_fmask()
        # subprocess.call([sys.executable,  script], shell=True)
        # DL.merge_tiles(directory=os.path.join(base_path, "data", "non_water_mask"), mode="clouds")
        poly = read_geojson(os.path.join(base_path, "poly.geojson"))
        DL.crop_non_water_mask(poly, "epsg:32616")
        DL.mask_predictions()

    if args.command == "clean":
        clean_directories(date=args.date[0])