from fmask.cmdline import sentinel2Stacked
import os
from utils.dir_management import base_path

os.environ['PROJ_LIB'] = '/home/henry/PycharmProjects/plastic_pipeline/plastic_proj/lib/python3.8/site-packages/pyproj/proj_dir/share/proj'
os.environ['PROJ_DEBUG'] = "3"


def run_fmask(path):
    safe_files = os.listdir(path)
    print("creating cloud mask...")
    for safe_file in safe_files:
        safe_file_path = os.path.join(path, safe_file)
        tile_id = safe_file.split("_")[-2]
        date = safe_file.split("_")[2][:8]
        out_img_file = f'{tile_id}_{date}_cloud.tif'
        out_dir = os.path.join(base_path, "data", "merged_geotiffs", out_img_file)
        sentinel2Stacked.mainRoutine(
            ['--safedir', safe_file_path, '-o', out_dir, '--tempdir', os.path.join(base_path, "fmask_api", "fmask_temp")])
        print(f'Completed running Fmask on SAFE file, saved as: {out_img_file}')
