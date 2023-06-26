import os, sys
import logging
import auxiliar as aux

# logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def reconstruct(files_merged_dl1, obs_id_LST, output_dir, config, process_data = True):

    # in the case the input is not an array (only an integer), we convert it to an array
    if (type(obs_id_LST)==int) or (type(obs_id_LST)==float):
        
        logger.info(f'Imput only one run: {obs_id_LST}')
        logger.debug(f'Converted {obs_id_LST} to a list format {[int(obs_id_LST)]}')
        obs_id_LST = [int(obs_id_LST)]
        
    elif (type(obs_id_LST)!=list):
        logger.warning(f'Error: invalid format for "obs_id_LST", needed an array of runs, or a unique run value') 
        sys.exit()
            
    # empty array for the filenames
    files_stereo_dl1 = []

    # importing only if needed
    if process_data == True:
        from magicctapipe.scripts import stereo_reconstruction
    else:
        logger.info(f'Stereo reconstruction already done, only extracting filenames (process_data=False)')

    # creating the output directory if do not exist
    aux.createdir(output_dir)

    # iterating over all LST runs
    for r in range(len(files_merged_dl1)):

        tmp_files_run = []
        
        output_dir_run = os.path.join(output_dir, f'Run{obs_id_LST[r]:05}')
        aux.createdir(output_dir_run)
        logger.debug(f'Output directory created: {output_dir_run}')
    
        # iterating over LST subruns
        for s in range(len(files_merged_dl1[r])):
            
            # run.subrun file from event coincidence
            input_file_lst  = files_merged_dl1[r][s]
            input_dir_magic = output_dir_run
            
            output_filename = f'dl1_stereo_LST-1_MAGIC.{input_file_lst[-16:]}' 

            # only finding the coincident events if we indicate that the process has not been done
            if process_data == True:
                stereo_reconstruction(input_file_lst, output_dir_run, config)
                
            tmp_files_run.append(os.path.join(output_dir_run, output_filename))

        files_stereo_dl1.append(tmp_files_run)

    return files_stereo_dl1
        