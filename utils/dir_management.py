import os
import shutil
import zipfile


# project directory
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data")
data_dirs = ["merged_geotiffs", "outputs", "patches", "predicted_patches", "processed", "unmerged_geotiffs", "unprocessed"]


def clean_dir(directory):
    for dir_path, _, filenames in os.walk(directory):
        for f in filenames:
            f_path = os.path.abspath(os.path.join(dir_path, f))
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


def setup_directories():
    for dir in data_dirs:
        os.makedirs(os.path.join(data_path, dir), exist_ok=True)
        dir_operation_selection(dir)


def clear_downloads():
    for dir in os.listdir(os.path.join(data_path, "downloads")):
        os.remove(os.path.join(data_path, "downloads", dir))


def breakdown_directories(date):
    # save outputs
    parent_dir = os.path.join(data_path, "outputs")
    output_path = os.path.join(parent_dir, date)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tiff_path = os.path.join(data_path, "merged_geotiffs")
    # only keep geotiff and final predictions
    for f in os.listdir(tiff_path):
        if not f.endswith("cloud.tif"):
            shutil.move(os.path.join(tiff_path, f), os.path.join(output_path, f))
    # clean or delete dirs
    for dir in data_dirs:
        dir_operation_selection(dir)


# clean, delete or pass
def dir_operation_selection(dir):
    if dir == "outputs":
        pass
    elif dir == "unprocessed":
        delete_dir(os.path.join(data_path, "unprocessed"))
    else:
        clean_dir(os.path.join(data_path, dir))


def unzip_files(files, path):
    for file in files:
        zip_path = os.path.join(path, file)
        with zipfile.ZipFile(os.path.join(zip_path), 'r') as zip_ref:
            zip_ref.extractall(path)
            os.remove(zip_path)


# gets all files with tagfrom directory
def get_files(path, tag):
    all_files = []
    for (root, dirs, files) in os.walk(path, topdown=True):
        for f in files:
            if tag in f:
                file_path = os.path.join(root, f)
                all_files.append(file_path)
    return all_files
