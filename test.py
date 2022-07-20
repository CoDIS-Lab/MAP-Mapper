import os
from paths import base_path
import xml.etree.ElementTree as ET

# this is a function which gets the coordinate reference system from the sentinel-2 SAFE file.
# this is important for masking as the meta-data of the tif files must have the correct crs
import rioxarray
import xarray

def nc_to_geotiff():


    xds = xarray.open_dataset('/home/henry/PycharmProjects/plastic_pipeline/data/historic_files/S2A_MSI_2022_05_03_21_19_31_T04QDK_L2R.nc')
    xds.rio.write_crs("epsg:32604", inplace=True)
    # xds["Evapotranspiration"].rio.to_raster('D:\Weather data\test.tif')

nc_to_geotiff()