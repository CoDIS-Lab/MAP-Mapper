import os
import shutil

from utils.paths import base_path


def setup_directories():
    data_path = os.path.join(base_path, "data")

    clean_dir(os.path.join(data_path, "patches"))

    clean_dir(os.path.join(data_path, "predicted_patches"))

    clean_dir(os.path.join(data_path, "processed"))

    delete_dir(os.path.join(data_path, "unprocessed"))

    clean_dir(os.path.join(data_path, "unmerged_geotiffs"))

    clean_dir(os.path.join(data_path, "merged_geotiffs"))


def clean_dir(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            f_path = os.path.abspath(os.path.join(dirpath, f))
            os.remove(f_path)


def delete_dir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def clean_directories(date):
    data_path = os.path.join(base_path, "data")

    clean_dir(os.path.join(data_path, "patches"))

    clean_dir(os.path.join(data_path, "predicted_patches"))

    clean_dir(os.path.join(data_path, "processed"))

    delete_dir(os.path.join(data_path, "unprocessed"))

    clean_dir(os.path.join(data_path, "unmerged_geotiffs"))

    parent_dir = os.path.join(data_path, "historic_files")
    historic_path = os.path.join(parent_dir,  date)
    if not os.path.exists(historic_path):
        os.mkdir(historic_path)
    tiff_path = os.path.join(data_path, "merged_geotiffs")
    # only keep geotiff and final predictions
    for f in os.listdir(tiff_path):
        if f.endswith(date+".tif") or f.endswith("prediction.tif"):
            shutil.move(os.path.join(tiff_path, f), os.path.join(historic_path, f))
        else:
            os.remove(os.path.join(tiff_path, f))
