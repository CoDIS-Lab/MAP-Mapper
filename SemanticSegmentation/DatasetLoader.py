import os

import numpy as np
import rasterio
from rasterio import windows
from itertools import product
from rasterio.merge import merge
from paths import base_path
from rasterio.plot import show
merged_path = os.path.join(base_path, "data", "merged_geotiffs")


def hex_2_rbg(hex):
    return np.array(tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4)))

def colour_map(self):
    # 1: 'Marine Debris',
    marine_debris = hex_2_rbg("ff0000")
    # 2: 'Dense Sargassum',
    dense_sargassum = hex_2_rbg("009900")
    # 3: 'Sparse Sargassum',
    sparse_sargassum = hex_2_rbg("009933")
    # 4: 'Natural Organic Material',
    organic = hex_2_rbg("006600")
    # 5: 'Ship',
    ship = hex_2_rbg("ffff00")
    # 6: 'Clouds',
    cloud = hex_2_rbg("#ffffff")
    # 7: 'Marine Water',
    water = hex_2_rbg("#0000ff")
    # 8: 'Sediment-Laden Water',
    sediment_water = hex_2_rbg("#cc9900")
    # 9: 'Foam',
    foam = hex_2_rbg("ffccff")
    # 10: 'Turbid Water',
    turbid_water = hex_2_rbg("996633")
    # 11: 'Shallow Water',
    shallow_water = hex_2_rbg("#3366cc")



class DatasetLoader:
    # def __init__(self,):
    #     self.processed_images =
    def load_images(self):
        # only get tif files from rhos (surface Level reflectence)
        self.tif_files = [os.path.join(base_path, "data", "processed", f) for f in os.listdir(os.path.join(base_path, "data", "processed")) if "rhos" in f and f.endswith(".tif")]
        #  makes sure each band is in correct order for stacking (splits off band wavelength number, removes ".tif", converts to int and sorts ascending)
        self.tif_files = sorted(self.tif_files, key=lambda x: int((x.rsplit("_", 1)[1][:-4])))

    def combine_images(self):
        split_tiles = {}
        for f in self.tif_files:
            # get tileid utilising file name format of the dataset. e.g(sensor_date_tileid_L2R_rhos_band.tif)
            f_name_as_list = f.split("_")
            tile_id = f_name_as_list[-4]
            date = "".join([x for x in f_name_as_list[3:6]])
            try:
                split_tiles[tile_id + "_" + date].append(f)
            except KeyError:
                split_tiles[tile_id + "_" + date] = [f]

        for key, value in split_tiles.items():
            # Read metadata of first file
            with rasterio.open(value[0]) as src0:
                meta = src0.meta
            # Update meta to reflect the number of layers
            meta.update(count = len(value))
            # Read each layer and write it to stack
            with rasterio.open(os.path.join(base_path, "data", "unmerged_geotiffs", key + '.tif'), 'w', **meta) as dst:
                for id, layer in enumerate(value, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(id, src1.read(1))

    def get_tiles(self, ds, width=256, height=256):
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    def merge_tiles(self, directory):
        tiff_files = os.listdir(directory)
        src_files_to_mosaic = []
        tile = tiff_files[1].split("_")
        date = tile[-1].strip(".tif")
        if date == "unet":
            date = tile[-2].strip(".tif")
        id = tile[0]
        merged_tile = id + "_" + date
        for fp in tiff_files:
            if fp.endswith(".tif"):
                src = rasterio.open(os.path.join(directory, fp))
                src_files_to_mosaic.append(src)
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        # with rasterio.open("/home/henry/PycharmProjects/plastic_pipeline/data/merged_geotiffs/T16PDC_T16PCC_T16QCD_T16QDD_20210918.tif", "r", **out_meta) as img:
        #     out_meta = img.meta.copy()
        out_meta.update({"driver": "GTiff",
                         })
        merged_image_path = os.path.join(merged_path, merged_tile + ".tif")
        with rasterio.open(merged_image_path, "w", **out_meta) as dest:
            dest.write(mosaic)

    def patch_image(self, input_filename, output_filename='tile_{}-{}.tif', in_path=base_path,
                    out_path=os.path.join(base_path, "data", "patches")):
        # https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
        with rasterio.open(os.path.join(in_path, input_filename)) as inds:
            tile_width, tile_height = 256, 256

            meta = inds.meta.copy()

            for window, transform in self.get_tiles(inds, width=256, height=256):
                print(window)
                if window.width == 256 and window.height == 256:
                    meta['transform'] = transform
                    meta['width'], meta['height'] = window.width, window.height
                    outpath = os.path.join(out_path, output_filename.format(int(window.col_off), int(window.row_off)))
                    with rasterio.open(outpath, 'w', **meta) as outds:
                        outds.write(inds.read(window=window))
                else:
                    print(str(window.width) + "X" + str(window.height) + " cropping and ignoring file")


    def run(self):
        # self.load_images()
        # self.combine_images()
        self.merge_tiles(directory=os.path.join(base_path, "data", "predicted_unet"))
       # self.merge_tiles(directory=os.path.join(base_path, "data", "unmerged_geotiffs"))
       #  for image in [image for image in os.listdir(merged_path)]:
       #      date = image.split("_")[-1][:-4]
       #      id = image.split("_")[0]
       #      self.patch_image(input_filename=os.path.join(merged_path, image), output_filename=id+"_{}_{}_"+date+".tif")


if __name__ == '__main__':
    DatasetLoader().run()

    