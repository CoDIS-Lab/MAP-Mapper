from flask import render_template
import pandas as pd
import json
import plotly
import os
import plotly.figure_factory as ff
from deployment import app
#from data import fetch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mapbox_token = os.environ.get('MAPBOX')
# plotly_params
hex_bin_number = 6  # higher number results in smaller and more numerous hex_bins
hex_bin_opacity = 0.5  # 0 (invisible) to 1 (opaque)
data_marker_size = 2  # plotted plastic detection marker size
data_marker_opacity = 0.4  # 0 (invisible) to 1 (opaque)
min_detection_count = 0  # lowest number of plastic detections needed for a hex_bin to be visualised on the map
max_plastic_percentage = 0.1 # remove all dates
close_to_land_mask = 2 # mask detections close to land (in km)
land_blur = 0.008999*close_to_land_mask
#excessive fmasking indicates scene is highly impacted by clouds or sunglint, removes dates with high mask percentage
max_masking_percentage = 80
min_depth = 5
port_mask_distance = 2500
csv_path = ''

@app.route('/', methods=['GET'])
def dashboard():
    #df = fetch()
    df = pd.read_csv(csv_path)
    fig = ff.create_hexbin_mapbox(
        title="Plastic Detections for ",
        data_frame=df, lat="latitude", lon="longitude",
        nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
        show_original_data=True,
        original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
        color_continuous_scale="Reds", min_count=min_detection_count,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=40, l=0, r=0))
 
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("map.html", graph_json=graph_json,)

@app.route('/view_data')
def view_database():
    #df = fetch()
    df = pd.read_csv(csv_path)
    return render_template("data.html", tables=[df.to_html(classes='data')], titles=df.columns.values)