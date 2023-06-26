import os, sys

import auxiliar as aux

# --- logging --- #
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def dl1_to_dl2(obs_id_LST, files_stereo_dl1, RFs_dir, output_dir, process_data):

    # empty array for saving the dl2 filenames
    files_dl2 = []
    
    # importing only if needed
    if process_data == True:
        from magicctapipe.scripts.lst1_magic import lst1_magic_dl1_stereo_to_dl2
    else:
        logger.info(f'dl1 --> dl2 already done, only extracting filenames (process_data=False)')
        
        
    # iterating over runs (normally this program should run for only one run)
    for r in range(len(files_stereo_dl1)):

        # create the directory for the dl2 files if do not exist
        aux.createdir(output_dir)  
        
        # output filename
        output_file = os.path.join(output_dir, f'dl2_LST-1_MAGIC.Run{obs_id_LST[r]:05}.h5')

        # only converting if we indicate that the process has not been done
        if process_data == True:  

            lst1_magic_dl1_stereo_to_dl2.dl1_stereo_to_dl2(files_stereo_dl1[r], RFs_dir, output_dir)

        files_dl2.append(output_file)

    return files_dl2