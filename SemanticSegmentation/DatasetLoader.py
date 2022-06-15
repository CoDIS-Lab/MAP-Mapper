import os

import geojson

import numpy as np
import pyproj
from shapely import ops
import rasterio
import shapely
import shapely.geometry
from rasterio import windows
from itertools import product
from rasterio.merge import merge
from paths import base_path
import glob
import shutil
import rasterio.mask
from osgeo import gdal
from osgeo import ogr
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


def project_wsg_shape_to_csr(shape, csr):
    project = lambda x, y: pyproj.transform(
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(init=csr),
        x,
        y
    )
    return shapely.ops.transform(project, shape)


class DatasetLoader:
    # cnsider holding images / variables in class variables to improve efficiency
    def __init__(self,):
        self.date = ""
        self.id = ""

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

    def merge_tiles(self, directory, mode):
        src_files_to_mosaic = []
        # merge predictions
        type = ""
        if mode == "masks":
            tiff_files = [x for x in os.listdir(directory) if x.endswith(".tif")]
            tile = tiff_files[1].split("_")
            self.id = tile[0]
            self.date = tile[-2].strip(".tif")
            type = "_unet"
            for fp in tiff_files:
                if fp.endswith(".tif"):
                    src = rasterio.open(os.path.join(directory, fp))
                    src_files_to_mosaic.append(src)
            mosaic, out_trans = merge(src_files_to_mosaic)
            # select first image from merge_geotiffs directory (must be kept with only one image to work properly)
            hyperspectral_geotiff = os.listdir(os.path.join(base_path, "data", "merged_geotiffs"))[0]
            # date is last 8 chars, followed by .tif
            with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", hyperspectral_geotiff), "r") as img:
                out_meta = img.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "count": 1,
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans
                             })
        # for merging all images into one image
        if mode == "images":
            tiff_files = [x for x in os.listdir(directory) if x.endswith(".tif")]
            tile = tiff_files[1].split("_")
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
        # merge predictions
        if mode == "flags":
            tiff_files = [x for x in os.listdir(directory) if x.endswith(".tif") and "flags" in x]
            # get tile id of first tile to use for reference
            tile = tiff_files[1].split("_")
            id = tile[-4]
            self.date = tile[2] + tile[3] + tile[4]
            type = "_flag"
            for fp in tiff_files:
                if fp.endswith(".tif"):
                    src = rasterio.open(os.path.join(directory, fp), "r+", nodata=99)
                    src.nodata = 99
                    src_files_to_mosaic.append(src)
            mosaic, out_trans = merge(src_files_to_mosaic)
            # select first image from merge_geotiffs directory (must be correct tile_id and date to work corredctly)
            hyperspectral_geotiff = [x for x in os.listdir(os.path.join(base_path, "data", "merged_geotiffs")) if x.endswith("_" + self.date + ".tif")][0]
            # date is last 8 chars, followed by .tif
            with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", hyperspectral_geotiff), "r") as img:
                out_meta = img.meta.copy()
                out_meta.update({"driver": "GTiff",
                                 "dtype": "int32",
                                 "count": 1,
                                 "nodata" : 99,
                                 "height": mosaic.shape[1],
                                 "width": mosaic.shape[2],
                                 "transform": out_trans
                                 })
        if mode == "clouds":
            #consider -load in meta date from safe file with eqch image, complete transform.. etc...
            tiff_files = [x for x in os.listdir(directory) if x.endswith(".tif") and "cloudresample" in x]
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
                             "transform":out_trans})



        merged_tile = self.id + "_" + self.date + type
        merged_image_path = os.path.join(merged_path, merged_tile + ".tif")
        with rasterio.open(merged_image_path, "w", **out_meta) as dest:
            dest.write(mosaic)






    def crop_non_water_mask(self, polygon, crs):
        projected_shape = project_wsg_shape_to_csr(shapely.geometry.shape(polygon), crs)

        with rasterio.open("/home/henry/PycharmProjects/plastic_pipeline/data/merged_geotiffs/T16QCD_20210918_cloud.tif") as dataset:
            out_image, out_transform = rasterio.mask.mask(dataset, [projected_shape], crop=True)
            out_meta = dataset.meta.copy()
            out_meta.update({"transform": out_transform, "height":out_image.shape[1], "width":out_image.shape[2]})
            with rasterio.open("/home/henry/PycharmProjects/plastic_pipeline/data/merged_geotiffs/T16QCD_20210918_cloud_crop.tif", "w", **out_meta) as src:
                src.write(out_image)

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
       # self.merge_tiles(directory=os.path.join(base_path, "data", "predicted_unet"), mode="masks")
       #self.merge_tiles(directory=os.path.join(base_path, "data", "unmerged_geotiffs"), mode="images")
       # self.merge_tiles(directory=os.path.join(base_path, "data", "processed"), mode="flags")
       #self.merge_tiles(directory=os.path.join(base_path, "data", "non_water_mask"), mode="clouds")
       # poly ={"type": "Polygon","coordinates": [[
       #          [-87.243, 16.5],
       #          [-88.956, 16.5],
       #          [-88.956, 15.671],
       #          [-87.243, 15.671],
       #          [-87.243, 16.5]
       #          ]]}
       # self.crop_non_water_mask(poly, "epsg:32616")
       self.mask_predictions()
       #  for image in [image for image in os.listdir(merged_path)]:
       #      date = image.split("_")[-1][:-4]
       #      id = image.split("_")[0]
       #      self.patch_image(input_filename=os.path.join(merged_path, image), output_filename=id+"_{}_{}_"+date+".tif")


    def mask_predictions(self):
        stack = []
        with rasterio.open("/home/henry/PycharmProjects/plastic_pipeline/data/merged_geotiffs/T16PDC_20210918_unet.tif") as predictions:
            array_a = predictions.read(1)
            out_meta = predictions.meta.copy()
            with rasterio.open("/home/henry/PycharmProjects/plastic_pipeline/data/merged_geotiffs/T16QCD_20210918_cloud_crop.tif") as mask:
                array_b = mask.read(1)
                # only consider mask values == 5 (these are tagged as water by fmask
                mask = (array_b != 5)
                mask = mask[:array_a.shape[0], :array_a.shape[1]]
                array_b = array_b[:array_a.shape[0], :array_a.shape[1]]
                new_array = np.copy(array_a)
                # convert all non-water pixels to category 6 (clouds) this category isnot important for analysis and therefore differentiation between land and clouds are not needed
                new_array[mask] = 6
                with rasterio.open("/home/henry/PycharmProjects/plastic_pipeline/data/merged_geotiffs/T16PDC_20210918_unet2.tif","w", **out_meta) as predictions1:
                    predictions1.write(np.expand_dims(new_array, axis=0))




    def clean_directories(self):
        data_path = os.path.join(base_path, "data")
        patch_files = glob.glob(os.path.join(data_path, "patches"))
        for f in patch_files:
            os.remove(f)
        patch_files = glob.glob(os.path.join(data_path, "predicted_unet"))
        for f in patch_files:
            os.remove(f)
        patch_files = glob.glob(os.path.join(data_path, "processed"))
        for f in patch_files:
            os.remove(f)
        patch_files = glob.glob(os.path.join(data_path, "unprocessed"))
        for f in patch_files:
            os.remove(f)
        patch_files = glob.glob(os.path.join(data_path, "unmerged"))
        for f in patch_files:
            os.remove(f)
        parent_dir = os.path.join(data_path, "historic_files")
        directory = self.date
        historic_path = os.path.join(parent_dir, directory)
        if not os.path.exists(historic_path):
            os.mkdir(historic_path)
        tiff_path = os.path.join(data_path, "merged_geotiffs")
        for f in os.listdir(tiff_path):
            shutil.move(os.path.join(tiff_path, f), os.path.join(historic_path, f))

   # def apply_acolite_mask(self):
        # tiff_files = [x for x in os.listdir(os.path.join(base_path, "data", "processed") if x.endswith(".tif")]
        # src_files_to_mosaic = []
        # tile = tiff_files[1].split("_")
        # if predicted_unet:
        #     self.date = tile[-2].strip(".tif") + "_unet"
        #     for fp in tiff_files:
        #         if fp.endswith(".tif"):
        #             src = rasterio.open(os.path.join(directory, fp))
        #             src_files_to_mosaic.append(src)
        #     mosaic, out_trans = merge(src_files_to_mosaic)
        #     # select first image from merge_geotiffs directory (must be kept with only one image to work properly)
        #     hyperspectral_geotiff = os.listdir(os.path.join(base_path, "data", "merged_geotiffs"))[0]
        #     # date is last 8 chars, followed by .tif
        #     with rasterio.open(os.path.join(base_path, "data", "merged_geotiffs", hyperspectral_geotiff),
        #                        "r") as img:
        #         out_meta = img.meta.copy()
        #     out_meta.update({"driver": "GTiff",
        #                      "count": 1,
        #                      "height": mosaic.shape[1],
        #                      "width": mosaic.shape[2],
        #                      })
if __name__ == '__main__':
    DatasetLoader().run()

    