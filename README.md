<h1> Marine Plastic Mapper </h1>

<h2> Introduction </h2>

Marine Plastic Mapper is a tool for assessing marine macro-plastic density. To identify hotspots, It is designed as a complete pipeline for downloading, processing and plotting suspected plastic detections and is underpinned by on the MARIDA dataset.

Marine Plastic Mapper comprised of multiple components built into one pipeline and can be run via a command line interface.
These components consit of:

Data downloading - SciHub is queried for Sentinel-2 data which is then downloaded to the local machine.

Pre-processing - Sentinel Safe files are processed with ACOLITE to perform atmospheric correction and outputs are prepared for parsing to a semantic segmentation algorithm.

Prediction - Predictions are generated from the tiff files.

Masking - Clouds and cloud shadows are masked using F-mask. A land_mask is generated using the region of interest coordinates and applied to the predictions. 

Data visualisation - The pixel coordinates of plastic pixels were extracted and converted to coordinate reference system points, consisting of a longitude and latitude. These are plotted on a Map using Plotly and a hexagonal heat map is generated.

<h2> Installation </h2> 

Installation requires integration of ACOLITE and F-mask and correct directory structure to be successful. Therefore care should be taken to ensure this is completed correctly.

<h3> Directory Structure </h3> 
.
└── plastic_pipeline/ <br>
    ├── acolite-main/ <br>
    ├── acolite_api/ <br>
    ├── analysis/ <br>
    ├── data/ <br>
    ├── masking/ <br>
    ├── semantic_segmentation/ <br>
    │   ├── smooth_patches/ <br>
    │   └── unet/ <br>
    ├── sentinel_downloader/ <br>
    ├── utils/ <br>
    │   └── world_land/ <br>
    ├── poly.geojson <br>
    └── run.py <br>
