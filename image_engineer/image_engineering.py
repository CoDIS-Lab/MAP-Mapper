import os
from itertools import product
import rasterio
import rasterio.mask
from rasterio import windows
from rasterio.merge import merge
# from utils.dir_management import base_path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
merged_path = os.path.join(base_path, "data", "merged_geotiffs")
unmerged_path = os.path.join(base_path, "data", "unmerged_geotiffs")
window_size = 32


class ImageEngineer:
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
        self.land_mask = kwargs.get("land_mask")
        self.cloud_mask = kwargs.get("cloud_mask")
    # load processed acolite images

    def load_images(self):
        # s2a and s2b wavelengths for bands 4, 6, 8, 11
        bands = ["665", "739", "833", "1614", "1610", "740"]
        print("getting rhos tiff files from acolite processed dir...")
        # only get tif files from rhos (surface Level reflectance)
        self.tif_files = [os.path.join(base_path, "data", "processed", f) for f in
                          os.listdir(os.path.join(base_path, "data", "processed")) if any(band in f for band in bands) and
                          "rhos" in f and f.endswith(".tif")]
        # makes sure each band is in correct order for stacking (splits off band wavelength number,
        # removes ".tif", converts to int and sorts ascending)
        self.tif_files = sorted(self.tif_files, key=lambda x: int((x.rsplit("_", 1)[1][:-4])))
        if self.tif_files:
            print("AC files found: " + str(self.tif_files))

    def combine_bands(self):
        print("Creating multi-banded tiff...")
        split_tiles = {}
        for f in self.tif_files:
            # get tile_id from acolite filename
            tile_id = f.split("_")[-4]
            try:
                split_tiles[tile_id + "_" + self.date].append(f)
            except KeyError:
                split_tiles[tile_id + "_" + self.date] = [f]

        for key, value in split_tiles.items():
            # Read metadata of first file
            with rasterio.open(value[0]) as src0:
                meta = src0.meta
            # Update meta to reflect the number of layers
            meta.update(count=4)
            # Read each layer and write it to stack
            with rasterio.open(os.path.join(base_path, "data", "unmerged_geotiffs", key + '.tif'), 'w', **meta) as dst:
                for id, layer in enumerate(value, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(id, src1.read(1))

    def merge_tiles(self, directory, mode):
        print("merging tiles... " + "mode: " + mode)
        src_files_to_mosaic = []
        # merge predictions
        tiff_dir = "merged_geotiffs"
        if mode == "masks":
            output_type = "unet"
            suffix = "predict.tif"
        if mode == "probs":
            output_type = "probabilities"
            suffix = "probs.tif"
        if mode == "images":
            output_type = ""
            suffix = ".tif"
            tiff_dir = "unmerged_geotiffs"
        if mode == "clouds":
            output_type = "cloud"
            suffix = "cloud.tif"
        tiff_files = [x for x in os.listdir(directory) if x.endswith(suffix)]
        for fp in tiff_files:
            src = rasterio.open(os.path.join(directory, fp))
            src_files_to_mosaic.append(src)

        mosaic, out_trans = merge(src_files_to_mosaic)
        # select first image from merge_geotiffs directory (must be kept with only one image to work properly)
        multi_banded_geotiff = self.id + "_" + self.date + ".tif"
        # get meta from multi banded geotiff
        with rasterio.open(os.path.join(base_path, "data", tiff_dir, multi_banded_geotiff), "r") as img:
            out_meta = img.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "count": 1,
                         "nodata": 99,
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans
                         })
        # for merging all images into one image
        if mode == "images":
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans
                             })
        if mode == "masks":
            out_meta.update({"dtype": "uint8"})
        if output_type:
            output_type = "_" + output_type
        merged_tile = self.id + "_" + self.date + output_type
        merged_image_path = os.path.join(merged_path, merged_tile + ".tif")
        with rasterio.open(merged_image_path, "w", **out_meta) as dest:
            dest.write(mosaic)

    # code to patch multi-banded GeoTiff by user2856
    # available @ https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
    # accessed 01/08/22
    def get_tiles(self, ds, width=window_size, height=window_size):
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

    # end of referenced code

    def patch_image(self, input_filepath, output_filename='{}_{}-{}_{}', in_path=base_path,
                    out_path=os.path.join(base_path, "data", "patches")):
        print("patching image @ " + input_filepath)
        with rasterio.open(os.path.join(in_path, input_filepath)) as inds:
            filename = input_filepath.split("/")[-1]
            tile = filename.split("_")[0]
            date = filename.split("_")[-1]
            # patch sizes are dependent on input image size
            if inds.width > 8000 and inds.height > 8000:
                tile_width, tile_height = inds.width // 16, inds.height // 16
            elif inds.width > 4000 and inds.height > 4000:
                tile_width, tile_height = inds.width // 8, inds.height // 8
            elif inds.width > 2000 and inds.height > 2000:
                tile_width, tile_height = inds.width // 4, inds.height // 4
            elif inds.width > 1000 and inds.height > 1000:
                tile_width, tile_height = inds.width // 2, inds.height // 2
            else:
                tile_width, tile_height = inds.width, inds.height
            meta = inds.meta.copy()
            for window, transform in self.get_tiles(inds, width=tile_width, height=tile_height):
                print(window)
                if window.width == tile_width and window.height == tile_height:
                    meta['transform'] = transform
                    meta['width'], meta['height'] = window.width, window.height
                    outpath = os.path.join(out_path, output_filename.format(tile, int(window.col_off), int(window.row_off), date))
                    with rasterio.open(outpath, 'w', **meta) as outds:
                        outds.write(inds.read(window=window))
                else:
                    print(str(window.width) + "X" + str(window.height) + " cropping and ignoring file")
