import numpy as np
import os, sys
import logging

import auxiliar as aux

# logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

#############################
# --- magic dl1 merging --- #
#############################
def MAGIC_dl1(obs_id_MAGIC, files_M1_dl1, files_M2_dl1, input_dir, output_dir, process_data = True):
    
    '''
    Input
    ------------
    --obs_id_MAGIC: 
            MAGIC runs to analyse
    --files_M1_dl2: 
            dl1 from MAGIC 1
    --files_M2_dl2: 
            dl1 from MAGIC 2
    --output_dir: 
            main output directory
    --input_dir: 
            main input directory
    --process_data: 
            If we want to process the data or only extract filenames (if processing was already done)
    Output:
    ------------
    --dl1_directories:
    
    '''
    
    # importing only if needed
    if process_data == True:
        from magicctapipe.scripts import merge_hdf_files
    else:
        logger.info(f'Merging already done, only extracting filenames (process_data=False)')

    # array to save the directories for the merged dl1 files
    files_MAGIC_dl1  = []

    # create the output directory if do not exists
    aux.createdir(output_dir)    
    # and create a directory for all the MAGIC runs of one night
    night_str = ','.join([str(i) for i in obs_id_MAGIC])
    output_dir_night = os.path.join(output_dir, night_str)
    
        
    # iterate over runs
    for r in range(len(files_M1_dl1)):             
        
        # temporal array for each run
        files_run = []

        # input and output directories
        input_fname    = f'dl1_M*.{files_M1_dl1[r][-9:]}'
        input_dir_run  = os.path.join(input_dir, f'Run{obs_id_MAGIC[r]:08}')
        output_fname   = files_M1_dl1[r][-21:].replace('M1','MAGIC').replace('M2','MAGIC')
        
        input_file     = os.path.join(input_dir_run,  input_fname)
        output_file    = os.path.join(output_dir_night, output_fname)

        logger.debug(f'Input file for run {obs_id_MAGIC[r]} :{input_file}')
        logger.debug(f'As output file :{output_file}')
            
        # merging
        # only if files not merged already
        if process_data == True:

            merge_hdf_files(input_dir_run, output_dir_night, run_wise=True)

        # saving the filename
        files_MAGIC_dl1.append(output_file)
    
    logger.info(f'Merged files by runs to {len(files_MAGIC_dl1)}')
    return files_MAGIC_dl1


##############################
# --- stereo dl1 merging --- #
##############################
# merging stereo subrun files into one unique LST run .h5 file
def stereo_dl1(obs_id_LST, files_stereo_dl1, input_dir, output_dir, process_data=True):
    '''
    merge stereo dl1 files
    
    Input
    ------------
    --obs_id_LST:
    --files_stereo_dl1:
    --input_dir:
    --output_dir:
    --process_data:
    
    Output
    ------------
    --file_runs: 
            merged dl1 files

    '''    
    # in the case the input is not an array (only an integer), we convert it to an array
    if (type(obs_id_LST)==int) or (type(obs_id_LST)==float):
        
        logger.info(f'Imput only one run: {obs_id_LST}')
        logger.debug(f'Converted {obs_id_LST} to a list format {[int(obs_id_LST)]}')
        obs_id_LST = [int(obs_id_LST)]
        
    elif (type(obs_id_LST)!=list):
        logger.warning(f'Error: invalid format for "obs_id_LST", needed an array of runs, or a unique run value') 
        sys.exit()
        
        
    file_run = []

    # importing only if needed
    if process_data == True:
        from magicctapipe.scripts import merge_hdf_files
    else:
        logger.info(f'Merging already done, only extracting filenames (process_data=False)')

    # iterating over all runs
    for r in range(len(files_stereo_dl1)):
        
        # folfer name for each run (subrun-wise data)
        input_dir_run = os.path.join(input_dir,  f'Run{obs_id_LST[r]:05}')
        output_file_run = os.path.join(output_dir, f'dl1_stereo_LST-1_MAGIC.Run{obs_id_LST[r]:05}.h5')
        
        # merging
        # only if files not merged already
        if process_data == True:

            merge_hdf_files(input_dir_run, output_dir, run_wise=True)
            
        file_run.append(output_file_run)
    
    return file_run


##############################
# --- stereo dl2 merging --- #
##############################
def stereo_dl2(input_dir, output_dir):
    '''
    merge stereo dl2 files
    
    Input
    ------------
    --input_dir:
    --output_dir:
    
    Output
    ------------

    '''    
    from magicctapipe.scripts import merge_hdf_files
    
    merge_hdf_files(input_dir, output_dir)