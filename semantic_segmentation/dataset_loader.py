import os
import shutil
import subprocess
import sys
from itertools import product

import numpy as np
import pyproj
import rasterio
import rasterio.mask
import shapely
import shapely.geometry
from rasterio import windows
from rasterio.merge import merge
from sentinelsat import read_geojson
from shapely import ops

from paths import base_path
from semantic_segmentation.debris_predictor import predict_with_smooth_blending
from utils.dir_management import clean_dir, delete_dir, clean_directories

merged_path = os.path.join(base_path, "data", "merged_geotiffs")
unmerged_path = os.path.join(base_path, "data", "unmerged_geotiffs")


# This function converts geojson polygon format to a the correct crs to enable fmask to be transposed onto the geotiff
def project_wsg_shape_to_csr(shape, csr):
    project = lambda x, y: pyproj.transform(
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(init=csr),
        x,
        y
    )
    return shapely.ops.transform(project, shape)


class DatasetLoader:
    # consider holding images / variables in class variables to improve efficiency
    def __init__(self, **kwargs):
        self.safe_trans = None
        self.tif_files = None
        self.date = None
        self.id = None
        self.crs = None
        if kwargs.get("date"):
            self.date = kwargs.get("date")
        if kwargs.get("id"):
            self.id = kwargs.get("id")
        if kwargs.get("crs"):
            self.crs = kwargs.get("crs")

    # load processed acolite images
    def load_images(self):
        print("getting rhos tiff files from processed dir...")
        # only get tif files from rhos (surface Level reflectance)
        self.tif_files = [os.path.join(base_path, "data", "processed", f) for f in
                          os.listdir(os.path.join(base_path, "data", "processed")) if
                          "rhos" in f and f.endswith(".tif")]
        # makes sure each band is in correct order for stacking (splits off band wavelength number,
        # removes ".tif", converts to int and sorts ascending)
        self.tif_files = sorted(self.tif_files, key=lambda x: int((x.rsplit("_", 1)[1][:-4])))

        f_name_as_list = self.tif_files[0].split("_")
        self.id = f_name_as_list[-4]
        print("Set tile ID as " + self.id)
        self.date = f_name_as_list[3] + f_name_as_list[4] + f_name_as_list[5]
        print("Set date as " + self.date)

        if self.tif_files:
            print("AC files found: " + str(self.tif_files))

    def combine_bands(self):
        print("Creating multi-banded tiff...")
        split_tiles = {}
        for f in self.tif_files:
            # get tileid utilising file name format of the dataset. e.g(sensor_date_tileid_L2R_rhos_band.tif)
            f_name_as_list = f.split("_")
            if not self.id:
                self.id = f_name_as_list[-4]
            if not self.date:
                self.date = f_name_as_list[3] + f_name_as_list[4] + f_name_as_list[5]
            try:
                split_tiles[self.id + "_" + self.date].append(f)
            except KeyError:
                split_tiles[self.id + "_" + self.date] = [f]

        for key, value in split_tiles.items():
            # Read metadata of first file
            with rasterio.open(value[0]) as src0:
                meta = src0.meta
            # Update meta to reflect the number of layers
            meta.update(count=len(value))
            # Read each layer and write it to stack
            with rasterio.open(os.path.join(base_path, "data", "unmerged_geotiffs", key + '.tif'), 'w', **meta) as dst:
                for id, layer in enumerate(value, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(id, src1.read(1))

    def get_tiles(self, ds, width=256, height=256):
        print("creating ")
        nols, nrows = ds.meta['width'], ds.meta['height']
        self.crs = ds.crs
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(
                big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    def merge_tiles(self, directory, mode):
        print("merging tiles... " + "mode: " + mode)
        src_files_to_mosaic = []
        # merge predictions
        type = ""
        if mode == "masks":
            tiff_files = [x for x in os.listdir(directory) if x.endswith(".tif")]
            tile = tiff_files[0].split("_")
            if not self.id:
                self.id = tile[0]
            if not self.date:
                self.date = tile[-2].strip(".tif")
            type = "_unet"
            for fp in tiff_files:
                if fp.endswith(".tif"):
                    src = rasterio.open(os.path.join(directory, fp))
                    src_files_to_mosaic.append(src)
            mosaic, out_trans = merge(src_files_to_mosaic)
            # select first image from merge_geotiffs directory (must be kept with only one image to work properly)
            hyperspectral_geotiff = self.id + "_" + self.date + ".tif"
            # date is last 8 chars, followed by .tif
            with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", hyperspectral_geotiff), "r") as img:
                out_meta = img.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "count": 1,
                             "nodata": 99,
                             "dtype": "uint8",
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans
                             })
        # for merging all images into one image
        if mode == "images":
            tiff_files = [x for x in os.listdir(directory) if x.endswith(".tif")]
            tile = tiff_files[0].split("_")
            self.id = tile[0]
            self.date = tile[-1].strip(".tif")
            for fp in tiff_files:
                if fp.endswith(".tif"):
                    src = rasterio.open(os.path.join(directory, fp))
                    src_files_to_mosaic.append(src)
            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta = src.meta.copy()
            self.safe_trans = out_trans
            out_meta.update({"driver": "GTiff",
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans
                             })

        if mode == "clouds":
            # consider -load in meta date from safe file with eqch image, complete transform.. etc...
            tiff_files = [x for x in os.listdir(directory) if x.endswith(".tif") and "cloud" in x]
            # get tile id of first tile to use for reference
            type = "_cloud"
            self.date = tiff_files[0].split("_")[1]
            self.id = tiff_files[0].split("_")[0]
            for fp in tiff_files:
                src = rasterio.open(os.path.join(directory, fp))
                src_files_to_mosaic.append(src)

            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans})

        merged_tile = self.id + "_" + self.date + type
        merged_image_path = os.path.join(merged_path, merged_tile + ".tif")
        with rasterio.open(merged_image_path, "w", **out_meta) as dest:
            dest.write(mosaic)

    def crop_non_water_mask(self, polygon, crs):
        projected_shape = project_wsg_shape_to_csr(shapely.geometry.shape(polygon), crs)
        with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs",
                                        self.id + "_" + self.date + "_cloud.tif")) as dataset:
            out_image, out_transform = rasterio.mask.mask(dataset, [projected_shape], crop=True)
            out_meta = dataset.meta.copy()
            out_meta.update(
                {"transform": out_transform, "height": out_image.shape[1] * 2, "width": out_image.shape[2] * 2})
            with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs",
                                            self.id + "_" + self.date + "_cloud_cropped.tif"), "w", **out_meta) as src:
                src.write(out_image)

    def patch_large_image(self, input_filepath, output_filename='{}_{}-{}_{}', in_path=base_path,
                          out_path=os.path.join(base_path, "data", "large_patches")):
        print("patching image @ " + input_filepath)
        # https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
        with rasterio.open(os.path.join(in_path, input_filepath)) as inds:
            filename = input_filepath.split("/")[-1]
            tile = filename.split("_")[0]
            date = filename.split("_")[-1]
            tile_width, tile_height = inds.width // 5, inds.height // 5

            meta = inds.meta.copy()

            for window, transform in self.get_tiles(inds, width=tile_width, height=tile_height):
                print(window)
                if window.width == tile_width and window.height == tile_height:
                    meta['transform'] = transform
                    meta['width'], meta['height'] = window.width, window.height
                    outpath = os.path.join(out_path,
                                           output_filename.format(tile, int(window.col_off), int(window.row_off), date))
                    with rasterio.open(outpath, 'w', **meta) as outds:
                        outds.write(inds.read(window=window))
                else:
                    print(str(window.width) + "X" + str(window.height) + " cropping and ignoring file")

    def mask_predictions(self):
        unet_file = [x for x in os.listdir(os.path.join(base_path, "data", "merged_geotiffs")) if "_unet" in x][0]
        with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", unet_file)) as predictions:
            prediction = predictions.read(1)
            out_meta = predictions.meta.copy()
            flag_file = \
                [x for x in os.listdir(os.path.join(base_path, "data", "merged_geotiffs")) if "_cloud_cropped" in x][0]
            with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", flag_file)) as mask_ds:
                # Only consider mask values == 5 (these are tagged as water by fmask).
                mask = (mask_ds.read(1) != 5)
                mask = mask[:prediction.shape[0], :prediction.shape[1]]
                masked_unet = np.copy(prediction)
                # Convert all non-water pixels to category 6 (clouds).
                # This category is not important for analysis.
                # Therefore, differentiation between land and clouds are not needed.
                masked_unet[mask] = 6
                with rasterio.open(
                        os.path.join(base_path, "data", "merged_geotiffs",
                                     self.id + "_" + self.date + "_unet_masked.tif"),
                        "w", **out_meta) as masked_prediction:
                    masked_prediction.write(np.expand_dims(masked_unet, axis=0))

    def run_pipeline(self):
        # load processed acolite rhos images (assigns file path to self.tiff_files)
        self.load_images()
        # combines processed satellite images output by acolite processor (one for each band)
        self.combine_bands()
        # merges sentinel 2 tiles into one large image covering whole region of interest
        # please note, this could be made more efficient by patching each tile, then merging the tiles.
        # However, care must be taken not to lose pixels due to cropping.
        # if using 1 or 2 sentinel tiles, it does not make much difference
        self.merge_tiles(directory=os.path.join(base_path, "data", "unmerged_geotiffs"), mode="images")
        # patch full ROI for predictions
        self.patch_large_image(os.path.join(base_path, "data", "merged_geotiffs", self.id + "_" + self.date + ".tif"))
        # make predictions on image patches
        predict_with_smooth_blending()
        # merge predicted masks into one file
        self.merge_tiles(directory=os.path.join(base_path, "data", "predicted_unet"), mode="masks")

        script = os.path.join(base_path, "Acolite", "fmask_script.py")
        subprocess.call([sys.executable, script])
        self.merge_tiles(directory=os.path.join(base_path, "data", "non_water_mask"), mode="clouds")

        poly = read_geojson(os.path.join(base_path, "poly.geojson"))
        self.crop_non_water_mask(poly, "epsg:32616")
        self.mask_predictions()
        clean_directories(self.date)



