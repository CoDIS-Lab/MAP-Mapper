import os
from paths import base_path
from fmask.cmdline import sentinel2Stacked
from multiprocessing import Pool
import rasterio

os.environ['PROJ_LIB'] = '/home/henry/PycharmProjects/plastic_pipeline/plastic_proj/lib/python3.8/site-packages/pyproj/proj_dir/share/proj'
#os.environ['GDAL_DATA'] = '/home/henry/anaconda3/envs/marida/share/gdal'
os.environ['PROJ_DEBUG'] = "3"


# this is used to with multiproceessing to increase speed,
# broken following CMD interface implementation (issue with subprocess)
def get_non_water_mask(safe_file):
    safe_file_path = os.path.join(path, safe_file)
    tile_id = safe_file.split("_")[-2]
    date = safe_file.split("_")[2][:8]
    out_img_file = f'{tile_id}_{date}_cloud.tif'
    out_dir = os.path.join(base_path, "data", "merged_geotiffs", out_img_file)
    sentinel2Stacked.mainRoutine(
        ['--safedir', safe_file_path, '-o', out_dir, '--tempdir', os.path.join(base_path, "acolite_api", "fmask_temp")])
    print(f'Completed running Fmask on SAFE file{out_img_file}')


print("running")
if __name__ == '__main__':
    path = os.path.join(base_path, "data", "unprocessed")
    safe_files = os.listdir(path)
    with Pool(10) as p:
        print(p.map(get_non_water_mask, safe_files))


def run_fmask():
    path = os.path.join(base_path, "data", "unprocessed")
    safe_files = os.listdir(path)
    for safe_file in safe_files:
        safe_file_path = os.path.join(path, safe_file)
        tile_id = safe_file.split("_")[-2]
        date = safe_file.split("_")[-1][:8]
        out_img_file = f'{tile_id}_{date}_cloud.tif'
        out_dir = os.path.join(base_path, "data", "merged_geotiffs", out_img_file)
        sentinel2Stacked.mainRoutine(
            ['--safedir', safe_file_path, '-o', out_dir, '--tempdir', os.path.join(base_path, "acolite_api", "fmask_temp")])
        print(f'Completed running Fmask on SAFE file, saved as: {out_img_file}')