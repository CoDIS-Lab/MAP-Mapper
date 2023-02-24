import os
from utils.dir_management import base_path
import acolite as ac


def run_acolite(bundle):
    # scenes to process
    # output directory
    odir = os.path.join(base_path, "data", "processed")
    # optional file with processing settings
    # if set to None defaults will be used
    settings_file = os.path.join(base_path, "acolite_api", "SETTINGS")
    # import settings
    settings = ac.acolite.settings.parse(settings_file)
    # set settings provided above
    settings['inputfile'] = os.path.join(base_path, "data", "unprocessed", bundle)
    settings['output'] = odir
    settings['polygon'] = os.path.join(base_path, "poly.geojson")

    # process the current bundle
    ac.acolite.acolite_run(settings=settings)
