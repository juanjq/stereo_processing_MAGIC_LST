import numpy as np
import sys
import yaml

sys.path.insert(0, '/fefs/aswg/workspace/juan.jimenez/stereo_analysis/scripts')

# --- logging --- #
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def stereo_dl1(obs_id_string, source_name):
    
    
    logger.info(f'Processing {source_name} data...')
    
    # if files aready processed to dl1 set to False
    # we can configure 1 by 1 if needed to process ony certain steps
    process = [True for _ in range(6)]
    
    process[0] = False    # converting MAGIC CAL to DL1
    process[1] = False    # merging M1 and M2
    process[2] = True     # event coincidence for the night
    process[3] = True     # stereo reconstruction
    process[4] = True     # merging the stereo files
    process[5] = False    # converting dl1 to dl2

    # configuration
    config_file = f'/fefs/aswg/workspace/juan.jimenez/stereo_analysis/config_files/config{source_name}.yaml'

    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)
        
    ###########################################
    ###########################################
    import find_files

    # run number of LST
    obs_id_LST, obs_id_MAGIC = find_files.str_to_run_ids(obs_id_string)

    # finding M1 and M2 callibrated directories
    files_M1_cal, files_M2_cal = find_files.find_MAGIC_cal(obs_id_MAGIC)

    # finding dl1 directories for LST
    files_LST_dl1 = find_files.find_LST_dl1(obs_id_LST)
    ###########################################
    ###########################################

    ###########################################
    ###########################################
    import magicruns_cal_to_dl1

    # main output directory
    output_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl1/runs_magic_m1_and_m2'

    files_M1_dl1 = magicruns_cal_to_dl1.convert(files_M1_cal, obs_id_MAGIC, output_dir, config, 1, process[0])
    files_M2_dl1 = magicruns_cal_to_dl1.convert(files_M2_cal, obs_id_MAGIC, output_dir, config, 2, process[0])   
    ###########################################
    ###########################################
    
    ###########################################
    ###########################################
    import merge

    input_dir  = '/fefs/aswg/workspace/juan.jimenez/data/dl1/runs_magic_m1_and_m2'
    output_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl1/nights_magic'

    files_MAGIC_dl1 = merge.MAGIC_dl1(obs_id_MAGIC, files_M1_dl1, files_M2_dl1, input_dir, output_dir, process[1])
    ###########################################
    ###########################################
    
    ###########################################
    ###########################################
    import event_coincidence_night

    output_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl1/joint_runs_magic_lst'

    files_coinc_dl1 = event_coincidence_night.coincidences(obs_id_LST, files_LST_dl1, files_MAGIC_dl1,
                                                            output_dir, config, process[2])
    ###########################################
    ###########################################
    
    ###########################################
    ###########################################
    import reco_stereo

    output_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl1/stereo_subruns'

    files_stereo_dl1_nomerged = reco_stereo.reconstruct(files_coinc_dl1, obs_id_LST, output_dir, config, process[3])
    ###########################################
    ###########################################
    
    input_dir  = '/fefs/aswg/workspace/juan.jimenez/data/dl1/stereo_subruns'
    output_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl1/stereo_runs'

    files_stereo_dl1 = merge.stereo_dl1(obs_id_LST, files_stereo_dl1_nomerged, input_dir, output_dir, process[4])
    ###########################################
    ###########################################
    
    
    ###########################################
    ###########################################    
    import convert_dl1_to_dl2

    if source_name == 'Crab':
        RFs_dir = '/fefs/aswg/LST1MAGIC/mc/models/ST0316A/NSB1.5/v01.2/dec_2276' # for CrabNebula
    elif source_name == 'BLLac':
        RFs_dir = '/fefs/aswg/LST1MAGIC/mc/models/ST0316A/NSB0.5/v01.2/dec_3476' # for BLLac
    elif source_name == 'Mrk421':
        RFs_dir = '/fefs/aswg/LST1MAGIC/mc/models/ST0316A/NSB0.5/v01.2/dec_4822' # for Mrk421
    else:
        logger.error(f'The source introduced {source_name} have not specified RFs or is bad written.')
        
    output_dir   = '/fefs/aswg/workspace/juan.jimenez/data/dl2/stereo_raw_dl2'
    
    files_stereo_dl2 = convert_dl1_to_dl2.dl1_to_dl2(obs_id_LST, files_stereo_dl1, RFs_dir, output_dir, process[5])    
    ###########################################
    ###########################################
    ###########################################
    
if __name__ == '__main__': 
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3])

