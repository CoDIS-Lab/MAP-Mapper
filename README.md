![BOH-0 5opacity3size0 99](https://user-images.githubusercontent.com/89064748/198129585-ca0e742e-4203-4897-a77c-488ec78e8654.png)


<h1> MAP-Mapper – User Guide </h1>

<h2> Edit on 13/October/2023 </h2>
<p> It has been brought to our attention that model.pth file was missing in the uploaded files. To solve this quickly, we uploaded the file online and you could download this here from the <a href="https://drive.google.com/file/d/1k9k3ansBUzj9EBoJqxGnakgStDMooob7/view?usp=sharing">link</a>

<h2> About </h2>
<p>Marine Plastic Mapper (MAP-Mapper) is a tool for assessing marine macro-plastic density to identify plastic hotspots, It is designed as a complete pipeline for downloading, processing and plotting suspected plastic detections and is underpinned by the MARIDA dataset.
</p>
<p>-&emsp;MAP-Mapper comprises of multiple components built into one pipeline and can be run via a command line interface. These components consist of:</p>
<p>-&emsp;Data downloading - The Open Access Corpernicus Hub is queried for relevant Sentinel-2 data which is then downloaded to the local machine.</p>
<p>-&emsp;Pre-processing - Sentinel -2 SAFE files are processed with ACOLITE to perform atmospheric correction. ACOLITE outputs are  then prepared for parsing to a semantic segmentation algorithm.</p>
<p>-&emsp;Prediction - Predictions are generated from the tiff files</p>
<p>-&emsp;Masking – Clouds and cloud shadows are masked using F-mask. Land is also masked.</p>
<p>-&emsp;Data visualisation - The pixel coordinates of plastic pixels were extracted and converted to coordinate reference system points, consisting of a longitude and latitude. These are plotted on a map and a hexagonal heat map is generated using Plotly. Detected plastic pixels are saved to a CSV file.</p>

<p> Please note, this install guide is for linux. This program can be made to run on Windows but it is likely that some small code changes will be needed to get it working.</p>
<h2> Setting up </h2>

1. Download acolite source code from  https://github.com/acolite/acolite and place acolite-main in the root directory <br><br>
2. To download Sentinel-2 data, you require an API username and password. This can be attained @ https://scihub.copernicus.eu/ <br>
These must be placed in a file named “.env” in the sentiner_downloader module like this: <br><br>
&emsp; USER_NAME=‘__API_username__’<br>
&emsp; PASSWORD=‘__API_password__’<br> <br>
3. To perform land masking, a worldwide land shape file is needed. <br>
Download the “large polygons not split” zip file from https://osmdata.openstreetmap.de/data/land-polygons.html <br>
unzip the folder and place all files in utils/world_land <br><br>

4. To run MAP-Mapper a valid geojson file must be placed in the root directory. This MUST be  called “poly.geojson”.
This file should contain a 4 sided convex polygon of the region you wish to map. This will consist of 5 sets of coordinates. The last set must be identical to the first. 
for example: <br>

&emsp;&emsp;{"type": "Polygon","coordinates": [[[120.2, 13.4], [121.1, 13.4], [121.1, 13.9], [120.2, 13.9], [120.2, 13.4]]]} <br>
<p> It is recommended that you do not use a polygon that exceeds the size of 4 sentinel-2 tiles. This is because the merged images become very large and slow to process. On some computers you may run the risk of running out of memory, causing the program to crash. If mapping a large region, map the area for one tile at a time and use non-overlapping polygons to ensure that the tiles do not overlap. </p>

<h2> Installing dependencies </h2>
conda env create -f env.yml

if getting an error with gdal use pip to install from wheel <br>

download from gdal from https://sourceforge.net/projects/gdal-wheels-for-linux/files/ (GDAL-3.4.1-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl) <br> <br>
Then run the commands: <br> <br>
&emsp; conda activate map-mapper <br>
&emsp; pip install /home/<user>/Downloads/GDAL-3.4.1-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl <br>
&emsp; pip install gdal-utils <br>

Finally, you must specify where the proj.db file is. If using anaconda this is likely to be in the shared folder of the MAP-Mapper environment. <br>
Navigate to run.py and change the environment variable to match this path. <br><br>
For example: <br>
&emsp; os.environ['PROJ_LIB'] = '/home/user/anaconda3/envs/map-mapper/share/proj'
<br>
<h2> Running </h2> 
To run the full pipeline, with cloud and land masking, navigate to the project root directory and use the terminal command:
	python   run.py   full   -start_date *   -end_date *   -cloud_mask   -land_mask
where * is a valid date of the format YYYYMMdd 

To specify a cloud percentage add: <br><br>
&emsp; -cloud_percentage * <br><br>
where * is a integer value between 0 and 100
The default cloud percentage is 20

if it recommended that you filter Sentinel-2 tiles by id. This ensures that only relevant  tiles are downloaded. It can also prevent the downloading of overlapping data, significantly improving processing time. <br>
To filter Sentinel-2 data by tiles add: <br><br>
&emsp; -tile_id * <br><br>
where * is a valid tile id, or list of tile ids, separated by a space. 
To identify the tile you require, Sentinel Hub can be useful. This is available @  https://apps.sentinel-hub.com/eo-browser/ 

<h2>  Analysis and Visualisation </h2>
To conduct analysis and visualisation. <br>
&emsp; cd /analysis  <br>
&emsp; run python analysis.py

<h2> Weather API </h2>
To filter Sentinel-2 data by wind speed, access to the visual crossing weather api is required.
This can be accessed for free @ https://www.visualcrossing.com/weather-api 
After signing up, an API key must be requested for their historical weather data. This enables 1000 free queries each day, which is far in excess of what is required for normal use.
The weather API key must be placed in the .env file in the sentiner_downloader module like this: <br><br>
&emsp;WEATHER=‘__weather_API_key__’ <br><br>
following this, a maximum wind speed can be set: <br><br>
&emsp; -max_wind * <br><br>
Where * is an interger value above 0.
