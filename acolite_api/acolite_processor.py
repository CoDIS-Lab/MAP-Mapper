import sys, os
from utils.dir_management import base_path
sys.path.insert(0, os.path.join(base_path, "acolite-main"))
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

    # process the current bundle
    ac.acolite.acolite_run(settings=settings)
