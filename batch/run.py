import argparse
from datetime import timedelta, datetime
import subprocess
from sentinel_downloader.sentinel_loader import SentinelLoader

if __name__ == "__main__":
     # set variables for downloading Sentinel-2 data
    today = datetime.today().strftime("%Y%m%d")
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y%m%d")

    # create argument parser for CLI
    parser = argparse.ArgumentParser(description='A sentinel-2 plastic detection pipeline using the MARIDA dataset')
    subparsers = parser.add_subparsers(help='possible uses', dest='command')

    # FULL MAP-MAPPER PIPELINE
    pipeline = subparsers.add_parser('full', help='run full pipeline for a given ROI')
    pipeline.add_argument(
        '-start_date',
        nargs=1,
        default=[today],
        type=str,
        help='start_date for sentinel 2 full_pipeline predictions to start (YYYYmmdd)',
        dest='start_date'

    )
    pipeline.add_argument(
        '-end_date',
        nargs=1,
        default=[tomorrow],
        type=str,
        help='end_date for sentinel 2 full_pipeline predictions to end (YYYYmmdd)',
        dest="end_date"
    )
    pipeline.add_argument(
        '-cloud_percentage',
        nargs=1,
        default=[20],
        type=int,
        help='maximum cloud percentage',
        dest="cloud_percentage"
    )
    args = parser.parse_args()
    products = SentinelLoader(start_date=args.start_date, end_date=args.end_date, max_cloud_percentage=args.cloud_percentage).get_product_data()
    # add code to get product ID
    for product_id in products:
        subprocess.run(["python", "run_job.py", f"-{product_id}"])