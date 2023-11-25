import numpy as np
import sys, os
import yaml

sys.path.insert(0, '/fefs/aswg/workspace/juan.jimenez/stereo_analysis/scripts')

import auxiliar as aux



def merge_nights(source_name):
    fname      = f'/fefs/aswg/workspace/juan.jimenez/stereo_analysis/config_files/common_data{source_name}.txt'
    dir_nights = f'/fefs/aswg/workspace/juan.jimenez/data/dl1/nights_magic/'

    subdirs = []
    with os.scandir(dir_nights) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name != '.ipynb_checkpoints':
                subdirs.append(entry.name)

    commondata = np.loadtxt(fname, dtype=str)
    night_strs = np.unique([f.split('-')[1] for f in commondata])

    magic_runs, night_correspondence = [], []
    for night_str in night_strs:
        for run in night_str.split(','):
            magic_runs.append(int(run))
            night_correspondence.append(night_str)


    # creating a folder for each night_string
    for night_str in night_strs:
        aux.createdir(os.path.join(dir_nights, night_str))

    # moving from unique folders to night folders, for each magic run
    for run, night_str in zip(magic_runs, night_correspondence):

        if os.path.exists(os.path.join(dir_nights, str(run))):

            source_folder      = os.path.join(dir_nights, str(run)) 
            destination_folder = os.path.join(dir_nights, night_str)
            aux.move_files(source_folder, destination_folder)

            # and deleting the directory after
            aux.delete_directory(source_folder)
    
    
if __name__ == '__main__': 
    merge_nights(sys.argv[1])

