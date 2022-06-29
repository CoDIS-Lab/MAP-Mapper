from lxml import etree
import numpy as np
import pandas as pd

# store the file path
xml_file = '/home/henry/PycharmProjects/plastic_pipeline/data/unprocessed/S2A_MSIL1C_20220429T161831_N0400_R040_T16PCC_20220429T200924.SAFE/GRANULE/L1C_T16PCC_A035787_20220429T163151/MTD_TL.xml'

# create a XML parser to parse the metadata and retrieve its root
parser = etree.XMLParser()
root = etree.parse(xml_file, parser).getroot()

# find the zenith node and the azimuth node for the sun angles using xpath expression
# as the result of xpath is always a list, we will index the first (and only) item
zenith_node = root.xpath('.//Sun_Angles_Grid/Zenith')[0]
azimuth_node = root.xpath('.//Sun_Angles_Grid/Azimuth')[0]

# read the zenith array
zenith_lst = zenith_node.xpath('.//VALUES/text()')
zenith_arr = np.array(list(map(lambda x: x.split(' '), zenith_lst))).astype('float')

# read the zenith array
azimuth_lst = azimuth_node.xpath('.//VALUES/text()')
azimuth_arr = np.array(list(map(lambda x: x.split(' '), azimuth_lst))).astype('float')

print(pd.DataFrame(zenith_arr).iloc[:5, :5], pd.DataFrame(azimuth_arr).iloc[:5, :5])
print("done")

import matplotlib.pyplot as plt

def get_grid_values_from_xml(tree_node, xpath_str):
    '''Receives a XML tree node and a XPath parsing string and search for children matching the string.
       Then, extract the VALUES in <values> v1 v2 v3 </values> <values> v4 v5 v6 </values> format as numpy array
       Loop through the arrays to compute the mean.
    '''
    node_list = tree_node.xpath(xpath_str)

    arrays_lst = []
    for node in node_list:
        values_lst = node.xpath('.//VALUES/text()')
        values_arr = np.array(list(map(lambda x: x.split(' '), values_lst))).astype('float')
        arrays_lst.append(values_arr)

    return np.nanmean(arrays_lst, axis=0)

sun_zenith = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Zenith')
sun_azimuth = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Azimuth')

view_zenith = get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Zenith')
view_azimuth = get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Azimuth')

_, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0,0].imshow(sun_zenith)
ax[0,0].set_title('Sun Zenith Angles')
ax[0,1].imshow(sun_azimuth)
ax[0,1].set_title('Sun Azimuth Angles')
ax[1,0].imshow(view_zenith)
ax[1,0].set_title('Viewing Zenith Angles')
ax[1,1].imshow(view_azimuth)
ax[1,1].set_title('Viewing Azimuth Angles')
plt.show()


def create_annotated_heatmap(hm, cmap='magma_r', vmin=None, vmax=None):
    '''Create an annotated heatmap. Parameter img is an optional background img to be blended'''
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.imshow(hm, vmin=vmin, vmax=vmax, cmap=cmap)

    # Loop over data dimensions and create text annotations.
    for i in range(0, hm.shape[0]):
        for j in range(0, hm.shape[1]):
            text = ax.text(j, i, round(hm[i, j], 2),
                           ha="center", va="center", color="cornflowerblue")

    return fig, ax


# convert angles arrays to radians
sun_zenith_rad = np.deg2rad(sun_zenith)
sun_azimuth_rad = np.deg2rad(sun_azimuth)

view_zenith_rad = np.deg2rad(view_zenith)
view_azimuth_rad = np.deg2rad(view_azimuth)

# calculate glint angle
phi = sun_azimuth_rad - view_azimuth_rad
Tetag = np.cos(view_zenith_rad) * np.cos(sun_zenith_rad) - np.sin(view_zenith_rad) * np.sin(sun_zenith_rad) * np.cos(
    phi)

# convert results to degrees
glint_array = np.degrees(np.arccos(Tetag))

fig, ax = create_annotated_heatmap(glint_array)
plt.show()
import rasterio
from pathlib import Path


def load_sen2cor_image(img_folder, bands):
    path = Path(img_folder)
    image = {}
    for band in bands:
        # considering the landsat images end with *_SR_B#.TIF, we will use it to locate the correct file
        file = next(path.rglob(f'*_{band}.jp2'))
        print(f'Opening file {file}')

        ds = rasterio.open(file)
        image.update({band: (ds.read(1) / 10000).astype('float32')})

    return image


# Create a RGB image
img_path = '/home/henry/PycharmProjects/plastic_pipeline/data/unprocessed/S2A_MSIL1C_20220429T161831_N0400_R040_T16PCC_20220429T200924.SAFE/GRANULE/L1C_T16PCC_A035787_20220429T163151/IMG_DATA'
img = load_sen2cor_image(img_path, ['B02', 'B03', 'B04'])

rgb = np.stack([img['B04'], img['B03'], img['B02']], axis=2)

plt.figure(figsize=(10, 10))
plt.imshow(rgb * 4)
plt.show()
plt.show()