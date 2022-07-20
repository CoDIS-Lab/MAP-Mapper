import rasterio
import numpy as np
from patchify import patchify, unpatchify
import math
import os
from paths import base_path
data_path = os.path.join(base_path, "data", "historic_files")
resolution = 100

def get_files_for_density_calc():
    density_files = []
    for (root, dirs, files) in os.walk(data_path, topdown=True):
        for f in files:
            if "unet_masked" in f:
                file_path = os.path.join(root, f)
                density_files.append(file_path)
    return density_files


def get_debris_coords(image_path):
    src = rasterio.open(image_path)
    image = src.read(1)
    debris_pixel_coords = np.argwhere(image == 1)
    return debris_pixel_coords


def get_debris_density(image_path):
    src = rasterio.open(image_path)
    image = src.read(1)
    debris_pixels = np.argwhere(image == 1)
    total_pixels = len(image) * len(image[0])
    total_void_pixels = len(np.argwhere(image == 6))
    density = len(debris_pixels) / (total_pixels - total_void_pixels)
    return density


def rounddown(x):
    return int(math.floor(x / resolution)) * resolution


def get_plastic_density(image_path, resolution):
    date = image_path.split("/")[-2]
    tile = image_path.split("/")[-1].split("_")[0]
    src = rasterio.open(image_path)
    image = src.read(1)
    meta = src.meta

    patches = patchify(image, (resolution, resolution), step=resolution) # split image into 100x100 small patches.
    patches = patches.astype("float32")
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
                patch = patches[i][j]
                debris_pixels = np.argwhere(patch == 1)
                total_pixels = len(patch) * len(patch[0])
                total_void_pixels = len(np.argwhere(patch == 6))
                if len(debris_pixels) > 0:
                    density = len(debris_pixels) / (total_pixels - total_void_pixels)
                elif total_pixels - total_void_pixels == 0:
                    density = np.NaN
                else:
                    density = 0
                density_patch = np.full(patch.shape, density)
                patches[i, j, :, :] = density_patch

    shape = image.shape
    shape = (rounddown(shape[0]), rounddown(shape[1]))
    meta.update({"dtype": "float32", "height":shape[0], "width":shape[1]})
    reconstructed_image = unpatchify(patches, shape)
    density_file_path = os.path.join(base_path, 'data', 'density_calc')
    density_image = os.path.join(density_file_path, tile + "_" + date + '_density.tif')

    with rasterio.open(density_image, 'w', **meta) as dst:
        dst.write(reconstructed_image.astype(rasterio.float32), 1)

#
# for file in get_files_for_density_calc():
#     get_plastic_density(file, resolution)



def get_longitudinal_plastic_density(dir_path):
    path_list = []
    stack = []
    for (root, dirs, files) in os.walk(dir_path, topdown=True):
        for f in files:
            image_path = os.path.join(root, f)
            path_list.append(image_path)
    for path in path_list:
        dataset = rasterio.open(path)
        image = dataset.read(1)
        stack.append(image)
    stack = np.array(stack)
    print(stack.shape)
    longitudinal_density_map = np.nanmean(stack, axis=0)
    # Read metadata of first file
    with rasterio.open(path_list[0]) as src0:
        meta = src0.meta

    with rasterio.open("longitudinal_density_map.tif", 'w', **meta) as dst:
        dst.write(longitudinal_density_map.astype(rasterio.float32), 1)

get_longitudinal_plastic_density("/home/henry/PycharmProjects/plastic_pipeline/data/density_calc")