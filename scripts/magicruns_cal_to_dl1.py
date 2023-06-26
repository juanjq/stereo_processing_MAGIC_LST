import os
import numpy as np
import numpy as nps
import auxiliar as aux
import logging

# logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# converting Callibration (.root) files to dl1 (.h5) for MAGIC
def convert(files_MAGIC_cal, obs_id_MAGIC, output_dir, config, tel_id , process_data = True):
    
    '''
    input:
    --files_MAGIC_cal: callibrated MAGIC filenames
    --id_MAGIC: IDs of MAGIC runs
    --output_dir: main output directory
    --config: configuration file
    --MAGIC_num: magi telecope ID (1,2)
    --process_data: If we want to process the data or only extract filenames (if processing was already done)
    output:
    -dl1 directories
    '''
    
    # not needed to import if data was already processed (takes a few seconds only)
    if process_data == True:
        from magicctapipe.scripts import magic_calib_to_dl1 
    else:
        logger.info(f'\ncalib --> dl1 already done, only extracting filenames (process_data=False)')
        
    # creating the output directory if do not exists
    aux.createdir(output_dir)
    
    # empty filename array
    files_MAGIC_dl1 = []
        
    # iterating over runs ---------
    for r in range(len(files_MAGIC_cal)):
        
        logger.info(f'\nConverting M{tel_id} run {obs_id_MAGIC[r]:08} to DL1...')      
        
        if len(files_MAGIC_cal[r]) != 0:
            # extracting the parent directory name for one MAGIC run, only for the first element readed
            input_dir_run    = os.path.dirname(files_MAGIC_cal[r][0])
            input_file_first = files_MAGIC_cal[r][0]
            output_dir_run   = os.path.join(output_dir, f'Run{obs_id_MAGIC[r]:08}')
            aux.createdir(output_dir_run)

            # IF DONE NOT COMPUTE ALL THE CALCULUS
            if process_data == True:

                # magic .root to .h5 script
                logger.debug(f'\nCalib --> DL1 input dir:{input_dir_run}')
                logger.debug(f'               output dir:{output_dir_run}')
                magic_calib_to_dl1(input_file_first, output_dir_run, config, process_run=True)
                logger.info( f'Converted Run{obs_id_MAGIC[r]:08} cal --> dl1')

            # output name of the dl1 files for one run
            output_file_run = os.path.join(output_dir_run, f'dl1_M{tel_id}.Run{obs_id_MAGIC[r]:08}.h5')

            files_MAGIC_dl1.append(output_file_run)

            logger.debug(f'For the MAGIC run {obs_id_MAGIC} created the file: {output_file_run}')
            logger.info(f'Converted {len(files_MAGIC_cal[r])} calibration files to a dl1 file')
        else:
            logger.warning(f'For MAGIC run: {obs_id_MAGIC[r]} no file is passed, passing to next one...')
    
    return files_MAGIC_dl1



# # converting Callibration (.root) files to dl1 (.h5) for MAGIC
# def convert(files_MAGIC_cal, obs_id_MAGIC, output_dir, config, tel_id , process_data = True):
    
#     '''
#     input:
#     --files_MAGIC_cal: callibrated bagic filenames
#     --id_MAGIC: IDs of MAGIC runs
#     --output_dir: main output directory
#     --config: configuration file
#     --MAGIC_num: magi telecope ID (1,2)
#     --process_data: If we want to process the data or only extract filenames (if processing was already done)
#     output:
#     -dl1 directories
#     '''
#     # not needed to import if data was already processed (takes a few seconds only)
#     if process_data == True:
#         from magicctapipe.scripts import magic_calib_to_dl1 
#     else:
#         logger.info(f'calib --> dl1 already done, only extracting filenames (process_data=False)')
        
#     # creating the output directory if do not exists
#     aux.createdir(output_dir)
    
#     # empty filename array
#     files_MAGIC_dl1 = []
        
#     # iterating over runs ---------
#     for r in range(len(files_MAGIC_cal)):
        
#         logger.info(f'\nConverting M{tel_id} run {obs_id_MAGIC[r]:08} to DL1...')


#         # create a directory for each run
#         output_dir_run = os.path.join(output_dir, f'{obs_id_MAGIC[r]}')
#         aux.createdir(output_dir_run)
#         logger.debug(f'Output directory created: {output_dir_run}')       
        
        
#         # extracting the parent directory name for one MAGIC run, only for the first element readed
#         input_dir_run    = os.path.dirname(files_MAGIC_cal[r][0])
#         input_file_first = files_MAGIC_cal[r][0]
        
#         # IF DONE NOT COMPUTE ALL THE CALCULUS
#         if process_data == True:

#             # magic .root to .h5 script
#             logger.debug(f'\nCalib --> DL1 input dir:{input_dir_run}')
#             logger.debug(f'               output dir:{output_dir_run}')
#             magic_calib_to_dl1(input_file_first, output_dir_run, config, process_run=True)
#             logger.info( f'Converted Run{obs_id_MAGIC[r]:08} cal --> dl1')
        
        
#         # temporal array for the run filenames
#         files_run = []        
        
#         # iterating over subruns only needed to extract the file name
#         for subrun in range(len(files_MAGIC_cal[r])):    
            
#             # finding the subrun
#             index_ref = files_MAGIC_cal[r][subrun].find(f'{obs_id_MAGIC[r]:08}')
#             subrun_id = int(files_MAGIC_cal[r][subrun][index_ref + 9 :index_ref + 12])
            
#             logger.debug(f'\nFor the run at path: {files_MAGIC_cal[r][subrun]} the identified subrun_id is: {subrun_id:03}')

#             output_filename = os.path.join(output_dir_run, f'dl1_M{tel_id}.Run{obs_id_MAGIC[r]:08}.{subrun_id:03}.h5')
#             files_run.append(output_filename)
#             logger.debug(f'Output file (without processing cal_to_dl1): {output_filename}')
            
#         files_MAGIC_dl1.append(files_run)
        
#         logger.info(f'Converted {len(files_MAGIC_cal[r])} calibration files to {len(files_run)} dl1 files')
    
#     return files_MAGIC_dl1
