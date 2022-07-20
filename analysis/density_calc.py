import os
from paths import base_path
data_path = os.path.join(base_path, "data", "historic_files")


def get_files_for_density_calc():
    density_files = []
    for (root, dirs, files) in os.walk(data_path, topdown=True):
        for f in files:
            if "unet_masked" in f:
                file_path = os.path.join(root, f)
                density_files.append(file_path)
    return density_files

