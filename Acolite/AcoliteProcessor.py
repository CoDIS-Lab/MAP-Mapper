import sys, os
user_home = os.path.expanduser("~")
sys.path.append(user_home+'/git/acolite')
import acolite as ac
from paths import base_path
def acolite_processor():
    # scenes to process
    bundles =  os.listdir(os.path.join(base_path, "data", "unprocessed"))
    # alternatively use glob
    # import glob
    # bundles = glob.glob('/path/to/scene*')

    # output directory
    odir = os.path.join(base_path, "data", "processed")

    # optional 4 element limit list [S, W, N, E]
    limit = None

    # optional file with processing settings
    # if set to None defaults will be used
    settings_file = os.path.join(base_path, "Acolite", "SETTINGS")

    # run through bundles
    for bundle in bundles:
        # import settings
        settings = ac.acolite.acolite_settings(settings_file)
        # set settings provided above
        settings['limit'] = limit
        settings['inputfile'] = bundle
        settings['output'] = odir
        # other settings can also be provided here, e.g.
        # settings['s2_target_res'] = 60
        # settings['dsf_path_reflectance'] = 'fixed'
        # settings['l2w_parameters'] = ['t_nechad', 't_dogliotti']

        # process the current bundle
        ac.acolite.acolite_run(settings=settings)

acolite_processor()