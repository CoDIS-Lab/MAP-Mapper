import os
import rasterio
from rasterio import windows
from itertools import product

from paths import base_path


class DatasetLoader:
    # def __init__(self,):
    #     self.processed_images =
    def load_images(self):
        # only get tif files from rhos (surface Level reflectence)
        self.tif_files = [os.path.join(base_path, "processed", f) for f in os.listdir(base_path, "processed") if "rhos" in f and f.endswith(".tif")]
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
                split_tiles[tile_id + date].append(f)
            except KeyError:
                split_tiles[tile_id + date] = [f]

        for key, value in split_tiles.items():
            # Read metadata of first file
            with rasterio.open(value[0]) as src0:
                meta = src0.meta
            # Update meta to reflect the number of layers
            meta.update(count = len(value))
            # Read each layer and write it to stack
            with rasterio.open(key + '.tif', 'w', **meta) as dst:
                for id, layer in enumerate(value, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(id, src1.read(1))

    # https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
    in_path = base_path
    input_filename = 'T16QCD20210918.tif'
    out_path = '/home/henry/Documents/Sentinel_Pipeline/patches'
    output_filename = 'tile_{}-{}.tif'

    def get_tiles(self, ds, width=256, height=256):
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    with rasterio.open(os.path.join(in_path, input_filename)) as inds:
        tile_width, tile_height = 256, 256

        meta = inds.meta.copy()

        for window, transform in get_tiles(inds):
            print(window)
            if window.width and window.height == 256:
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
                with rasterio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))
            else:
                print(str(window.width) + "X" + str(window.height) + " cropping and ignoring file")

    #def run(self):
    # self.load_images()
    # self.combine_images()
    #self.patch_image("/home/henry/Documents/Sentinel_Pipeline/T16QCD20210918.tif", 256)
  
if __name__ == '__main__':
    DatasetLoader().run()

    