import yaml
import sys

# path to the scripts
sys.path.insert(0, '/fefs/aswg/workspace/juan.jimenez/stereo_analysis/scripts')
import find_files

# --- logging --- #
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

###############################################
# checking if magic runs are in the directory #
###############################################
def check_sruns(obs_id):
    '''
    Input
    -------------
    - obs_id: int array
    
    Output
    -------------
    
    '''

    obs_id = [int(obs_id)]
    
    # finding M1 and M2 callibrated directories
    files_M1_cal, files_M2_cal = find_files.find_MAGIC_cal(obs_id, log=False)

    # checking number of subruns found
    return len(files_M1_cal[0])



##########################################################################
# convert one run of MAGIC from calibration to dl1, and merge M1 with M2 #
##########################################################################
def cal_dl1(obs_id, source_name):
    '''
    Input
    -------------
    - obs_id: int array
            
    
    Output
    -------------
    
    
    '''
    
    # configuration
    config_file = f'/fefs/aswg/workspace/juan.jimenez/stereo_analysis/config_files/config{source_name}.yaml'
    logger.info(f'Running for {source_name}...\n\nTaking the configuration file from --> {config_file}')

    # opening the configuration file
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)
        
    ###########################################
    ###########################################
    import find_files

    obs_id = [int(obs_id)]
    
    # finding M1 and M2 callibrated directories
    files_M1_cal, files_M2_cal = find_files.find_MAGIC_cal(obs_id)

    # checking if no MAGIC runs are found
    if (len(files_M1_cal[0]) == 0) or (len(files_M2_cal[0]) == 0):
        
        logger.error(f'MAGIC run passed is empty, i.e. not files found\n\n')
        sys.exit()
    
    ###########################################
    ###########################################

    ###########################################
    ###########################################
    import magicruns_cal_to_dl1

    # main output directory
    output_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl1/runs_magic_m1_and_m2'

    files_M1_dl1 = magicruns_cal_to_dl1.convert(files_M1_cal, obs_id, output_dir, config, 1)
    files_M2_dl1 = magicruns_cal_to_dl1.convert(files_M2_cal, obs_id, output_dir, config, 2)   
    ###########################################
    ###########################################
    
    ###########################################
    ###########################################
    import merge

    input_dir  = '/fefs/aswg/workspace/juan.jimenez/data/dl1/runs_magic_m1_and_m2'
    output_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl1/nights_magic'

    files_MAGIC_dl1 = merge.MAGIC_dl1(obs_id, files_M1_dl1, files_M2_dl1, input_dir, output_dir)
    ###########################################
    ###########################################  

    
                            
if __name__ == '__main__': 
    
    # the case of checking runs
    if globals()[sys.argv[1]] == check_sruns:
        nsruns = check_sruns(sys.argv[2])
        sys.exit(nsruns)  
    
    # the case of performing the step
    else:
        globals()[sys.argv[1]](sys.argv[2], sys.argv[3])
