import os, sys
import pandas as pd
import logging
import auxiliar as aux

from ctapipe.io import read_table

# logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# finding the MAGIC window for a whole night
def MAGIC_night_window(files_MAGIC_dl1):
    
    # time arrays for each MAGIC run
    tmin, tmax = [], []
    
    # iterating over all runs
    for m in range(len(files_MAGIC_dl1)):
        
        # opening run file
        event_source = pd.read_hdf(files_MAGIC_dl1[m],  key='events/parameters')

        tmin.append(event_source['time_sec'][0]       + 1e-9 * event_source['time_nanosec'][0])
        tmax.append(event_source['time_sec'].iloc[-1] + 1e-9 * event_source['time_nanosec'].iloc[-1])
        
        logger.debug(f'The time window of {files_MAGIC_dl1} is [{tmin[-1]},{tmax[-1]}]')
    
    return [tmin, tmax]

# finding the time window for a LST run.subrun
def LST_subrun_window(file_LST_dl1_run_subrun):
    
    event_source = read_table(file_LST_dl1_run_subrun, '/dl1/event/telescope/parameters/LST_LSTCam')

    tmin = event_source['dragon_time'][ 0]
    tmax = event_source['dragon_time'][-1]
    
    logger.debug(f'The time window of {file_LST_dl1_run_subrun} is [{tmin},{tmax}]')
    
    return [tmin, tmax]

# given 1 LST time window, and the multiple MAGIC time windows, returning if LST is in some MAGIC or not
def is_LST_in_MAGIC(LST_window, MAGIC_windows):
    
    mask_timewindows = []
    
    # for each MAGIC run we compare with a LST subrun for the time windows coincidences
    for m in range(len(MAGIC_windows[0])):
        mask_timewindows.append(((MAGIC_windows[1][m] > LST_window[0]) and (MAGIC_windows[0][m] < LST_window[1])))

    # if the coincidence with all MAGIC is false, coincide=False, if only one is true coincide=True
    coincide = False if sum(mask_timewindows) == 0 else True
    logger.debug(f'Coincidence is {coincide} and the number of MAGIC coincidences is {sum(mask_timewindows)}')
    
    return coincide


# finding the coincidences and generating the files
def coincidences(obs_id_LST, files_LST_dl1, files_MAGIC_dl1, output_dir, config, process_data=True, process_windows=True):
    
    # in the case the input is not an array (only an integer), we convert it to an array
    if (type(obs_id_LST)==int) or (type(obs_id_LST)==float):
        
        logger.info(f'Imput only one run: {obs_id_LST}')
        logger.debug(f'Converted {obs_id_LST} to a list format {[int(obs_id_LST)]}')
        obs_id_LST = [int(obs_id_LST)]
        
    elif (type(obs_id_LST)!=list):
        logger.warning(f'Error: invalid format for "obs_id_LST", needed an array of runs, or a unique run value') 
        sys.exit()
    
    # empty array for the filenames
    files_merged_dl1 = [] 

    # importing only if needed, and extracting time windows also if needed
    if process_data == True:
        from magicctapipe.scripts import event_coincidence
        
    else:
        logger.info(f'Event coincidence already done, only extracting filenames (process_data=False)')

    if process_windows == True:
        # extracting the timewindow of all MAGIC
        MAGIC_windows = MAGIC_night_window(files_MAGIC_dl1)
        
    # creating the output directory
    aux.createdir(output_dir)
        
    # iterating over LST runs
    for r in range(len(files_LST_dl1)):
        
        # creating the LST run directory
        output_dir_run = os.path.join(output_dir, f'Run{obs_id_LST[r]:05}')
        aux.createdir(output_dir_run)
        logger.debug(f'Output directory created: {output_dir_run}')        
        
        # temporal array for the data of the run
        tmp_data_run  = []
        
        # iterating over LST subruns
        for s in range(len(files_LST_dl1[r])):
            
            if process_windows == True:
                LST_window         = LST_subrun_window(files_LST_dl1[r][s])
                flag_LST_contained = is_LST_in_MAGIC(LST_window, MAGIC_windows)
                
                logger.debug(f'LST timewindow is [{int(LST_window[0])},{int(LST_window[1])}]')
                logger.debug(f'MAGIC timewindows; min_t = {[int(M) for M  in MAGIC_windows[0]]}')
                logger.debug(f'MAGIC timewindows; max_t = {[int(M) for M  in MAGIC_windows[1]]}') 
                
            else:
                # in the case data is already processes there is no need on computing the timewindows
                flag_LST_contained = True
            
            
            if flag_LST_contained:   
                
                # run.subrun file from LST
                # magic input files directory (merged subrun files)
                input_file_lst  = files_LST_dl1[r][s]
                input_dir_magic = os.path.dirname(files_MAGIC_dl1[0])

                output_filename = f'dl1_LST-1_MAGIC.{input_file_lst[-16:]}' 

                # only finding the coincident events if we indicate that the process has not been done
                if process_data == True:
                    
                    logger.debug(f'For LST subrun {s} there IS time coincidence with MAGIC runs')
                    event_coincidence(input_file_lst, input_dir_magic, output_dir_run, config)

                tmp_data_run.append(os.path.join(output_dir_run, output_filename))

                
            else:

                logger.warning(f'For LST run {obs_id_LST[r]} subrun {s} there is no time coincidence with MAGIC')
                
        files_merged_dl1.append(tmp_data_run)
        
    return files_merged_dl1


# # finding the MAGIC window for a whole night
# def MAGIC_night_window(files_MAGIC_dl1):
    
#     # time arrays for each MAGIC run
#     tmin, tmax = [], []
    
#     # iterating over all runs
#     for m in range(len(files_MAGIC_dl1)):
        
#         # firs cubrun file
#         event_source_init = pd.read_hdf(files_MAGIC_dl1[m][0],  key='events/parameters')
#         # last subrun file
#         event_source_last = pd.read_hdf(files_MAGIC_dl1[m][-1], key='events/parameters')

#         tmin.append(event_source_init['time_sec'][0]       + 1e-9 * event_source_init['time_nanosec'][0])
#         tmax.append(event_source_last['time_sec'].iloc[-1] + 1e-9 * event_source_init['time_nanosec'].iloc[-1])
        
#         logger.debug(f'The time window of {files_MAGIC_dl1} is [{tmin[-1]},{tmax[-1]}]')
    
#     return [tmin, tmax]

# # finding the time window for a LST run.subrun
# def LST_subrun_window(files_LST_dl1_run_subrun):
    
#     event_source = read_table(files_LST_dl1_run_subrun, '/dl1/event/telescope/parameters/LST_LSTCam')

#     tmin = event_source['dragon_time'][ 0]
#     tmax = event_source['dragon_time'][-1]
    
#     logger.debug(f'The time window of {files_LST_dl1_run_subrun} is [{tmin},{tmax}]')
    
#     return [tmin, tmax]

# # given 1 LST time window, and the multiple MAGIC time windows, returning if LST is in some MAGIC or not
# def is_LST_in_MAGIC(LST_window, MAGIC_windows):
    
#     mask_timewindows = []
    
#     # for each MAGIC run we compare with a LST subrun for the time windows coincidences
#     for m in range(len(MAGIC_windows[0])):
#         mask_timewindows.append(((MAGIC_windows[1][m] > LST_window[0]) and (MAGIC_windows[0][m] < LST_window[1])))

#     # if the coincidence with all MAGIC is false, coincide=False, if only one is true coincide=True
#     coincide = False if sum(mask_timewindows) == 0 else True
#     logger.debug(f'Coincidence is {coincide} and the number of MAGIC coincidences is {sum(mask_timewindows)}')
    
#     return coincide


# # finding the coincidences and generating the files
# def coincidences(obs_id_LST, files_LST_dl1, files_MAGIC_dl1, output_dir, config, process_data = True):
    
#     # in the case the input is not an array (only an integer), we convert it to an array
#     if (type(obs_id_LST)==int) or (type(obs_id_LST)==float):
        
#         logger.info(f'Imput only one run: {obs_id_LST}')
#         logger.debug(f'Converted {obs_id_LST} to a list format {[int(obs_id_LST)]}')
#         obs_id_LST = [int(obs_id_LST)]
        
#     elif (type(obs_id_LST)!=list):
#         logger.warning(f'Error: invalid format for "obs_id_LST", needed an array of runs, or a unique run value') 
#         sys.exit()
    
#     # empty array for the filenames
#     files_merged_dl1 = [] 

#     # importing only if needed
#     if process_data == True:
#         from magicctapipe.scripts import event_coincidence
#     else:
#         logger.info(f'Event coincidence already done, only extracting filenames (process_data=False)')

#     # creating the output directory for the night if do not exist
#     aux.createdir(output_dir)
    
#     night_string     = '-'.join([f'{i:05}' for i in obs_id_LST])
#     output_dir_night = os.path.join(output_dir, f'Run{night_string}')
#     aux.createdir(output_dir_night)
#     logger.debug(f'Output directory created: {output_dir_night}')

#     # iterating over LST runs
#     for r in range(len(files_LST_dl1)):

#         MAGIC_windows = MAGIC_night_window(files_MAGIC_dl1)
#         tmp_data_run  = []
        
#         # iterating over LST subruns
#         for s in range(len(files_LST_dl1[r])):
            
#             LST_window = LST_subrun_window(files_LST_dl1[r][s])
            
#             logger.debug(f'LST timewindow is [{int(LST_window[0])},{int(LST_window[1])}]')
#             logger.debug(f'MAGIC timewindows; min_t = {[int(M) for M  in MAGIC_windows[0]]}')
#             logger.debug(f'MAGIC timewindows; max_t = {[int(M) for M  in MAGIC_windows[1]]}')
            
            
#             if is_LST_in_MAGIC(LST_window, MAGIC_windows):   
                
#                 logger.info(f'For LST subrun {s} there IS time coincidence with MAGIC runs')
                
#                 # run.subrun file from LST
#                 # magic input files directory (merged subrun files)
#                 input_file_lst   = files_LST_dl1[r][s]
#                 input_dir_magic  = os.path.dirname(files_MAGIC_dl1[r][0])

#                 output_filename = f'dl1_LST-1_MAGIC.{input_file_lst[-16:]}' 

#                 # only finding the coincident events if we indicate that the process has not been done
#                 if process_data == True:

#                     event_coincidence(input_file_lst, input_dir_magic, output_dir_night, config)

#                 tmp_data_run.append(os.path.join(output_dir_night, output_filename))

                
#             else:

#                 logger.warning(f'For LST subrun {s} there is no time coincidence with MAGIC')
                
#         files_merged_dl1.append(tmp_data_run)
        
#     return files_merged_dl1
