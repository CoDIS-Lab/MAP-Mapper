import sys, os
from paths import base_path
import acolite as ac
from paths import base_path
from multiprocessing import Pool


def acolite_loader(bundle):
    # scenes to process
    # output directory
    odir = os.path.join(base_path, "data", "processed")

    # optional 4 element limit list [S, W, N, E]
    #limit = [15.671, -88.956, 16.5, -87.243]

    # optional file with processing settings
    # if set to None defaults will be used
    settings_file = os.path.join(base_path, "Acolite", "SETTINGS")
    # import settings
    settings = ac.acolite.settings.parse(settings_file)
    # set settings provided above
    #settings['limit'] = limit
    settings['inputfile'] = os.path.join(base_path, "data", "unprocessed", bundle)
    settings['output'] = odir
    # other settings can also be provided here, e.g.
    # settings['s2_target_res'] = 60
    # settings['dsf_path_reflectance'] = 'fixed'
    # settings['l2w_parameters'] = ['t_nechad', 't_dogliotti']

    # process the current bundle
    ac.acolite.acolite_run(settings=settings)


bundles = bundles = os.listdir(os.path.join(base_path, "data", "unprocessed"))

if __name__ == '__main__':
    with Pool(40) as p:
        print(p.map(acolite_loader, bundles))