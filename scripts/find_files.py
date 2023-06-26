import numpy as np
import os
import sys
import glob

# --- logging --- #
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


####################################################################
# function to convert a string with files to arrays of run numbers #
####################################################################
def str_to_run_ids(input_str):
    '''
    A function that converts the string that is extracted from 
    .txt files to arrays of run ids.
    
    Input
    ------------
    --input_str: str
            string in the format LST_runs-MAGIC_runs, all separated by commas
    
    Output
    ------------
    --obs_id_LST: int array
            array with the LST runs
    
    - obs_id_MAGIC: int array
            array with the MAGIC runs
    '''
    
    index_ref = input_str.find('-') # middle reference index

    # if '-' is not found, raise a warning
    if index_ref == -1:
        logger.error(f'Incorrect format "LSTrun-MAGICruns(comma separated)" is needed')
        sys.exit()

    # extracting run numbers
    obs_id_LST   = [int(i) for i in input_str[  :index_ref].split(',')]
    obs_id_MAGIC = [int(i) for i in input_str[index_ref+1:].split(',')]
    
    # diferentiating the cases where one or more LST runs are found
    if (len(obs_id_LST) > 1):
        logger.info(f'Given {len(obs_id_LST)} LST runs')
        return obs_id_LST, obs_id_MAGIC
        
    elif (len(obs_id_LST) == 1):
        logger.info(f'Given only {len(obs_id_LST)} LST run')
        return obs_id_LST, obs_id_MAGIC
    
    else:
        logger.error(f'Incorrect format "LSTruns-MAGICruns"(comma separated) is needed')
        sys.exit()
        
########################################
# ----- find .h5 files for MAGIC ----- #
########################################
# they are located at '/fefs/onsite/common/MAGIC/data/MX/event/Calibrated' inside MX=M1,M2 folders for MAGIC 1 and 2
# the output is a list of directories for each input run
def find_MAGIC_cal(obs_ids):
    
    '''
    Input
    ------------
    --obs_ids: int array
            array of MAGIC run ids
    
    Output
    ------------
    --filenames_M1: str array
            filenames for MAGIC-I
            
    --filenames_M2: str array
            filenames for MAGIC-II
    '''
    
    # --- defining the root paths --- #
    # root directory at IT cluster for MAGIC calibration files (.root)
    root_MAGIC = '/fefs/onsite/common/MAGIC/data/MX/event/Calibrated'  
    
    # main directory depending if we have MAGIC 1 or 2
    root = [root_MAGIC.replace('MX','M1'), root_MAGIC.replace('MX','M2')]
    logger.info(f'Input MAGIC-I  files path: {root[0]}')
    logger.info(f'Input MAGIC-II files path: {root[1]}')

    # both files for M1 and M2
    filenames = [[], []] # for all filenames
    files_run = [[], []] # for filtered filenames
    
    # --- file tree --- #
    # iterating over MAGIC 1 and 2 ---------
    logger.info(f'\nFinding all runs under all directories...')
    for m in [0,1]:
        
        # the directories have the following structure
        # 'root/year/month/day/***.root'
        
        # iterating over years ---------
        for year in [name for name in os.listdir(root[m]) if os.path.isdir(os.path.join(root[m], name))]:
            # year filename root/year
            fnameY = os.path.join(root[m], year)

            # iterating over months ---------
            for month in [name for name in os.listdir(fnameY) if os.path.isdir(os.path.join(fnameY, name))]:
                # month filename root/year/month
                fnameM = os.path.join(fnameY, month)

                # iterating over days ---------
                for day in [name for name in os.listdir(fnameM) if os.path.isdir(os.path.join(fnameM, name))]:
                    # day filename root/year/month/day
                    fnameD = os.path.join(fnameM, day)

                    # iterating over files in each day ---------
                    for f in os.listdir(fnameD):

                        # avoiding signal and other filenames
                        if ('signal' not in f) and ('.root' in f):

                            # appending every filename
                            filenames[m].append(os.path.join(fnameD,f))

        # selecting only the selected runs
        logger.info(f'\nFiltering the selected obs_ids...')
        for run in obs_ids:
            coincident_runs = np.array([file for file in filenames[m] if f'_{run:08}.' in file])
            files_run[m].append(coincident_runs)
            
            logger.info(f'For the MAGIC run {run:08} the number of subruns found are {len(coincident_runs)}')

        # sorting subruns to be in proper order
        for r in range(len(obs_ids)):

            # subruns array
            subruns = []
                
            # iterating over files-subrun
            for f in files_run[m][r]:

                # in order to extract the subrun of a file we find the run inside the filename
                ref_index = f.find(f'M{m+1}_')
                # and save the subrun number
                subruns.append(int(f[ref_index+12 : ref_index+15]))

            # sorted indexes    
            index_mask = np.array(subruns).argsort()
            # sorting the selected files
            files_run[m][r] = files_run[m][r][index_mask]

    # sometimes the number of runs of M1 and M2 are different
    # for avoiding this we will only keep the coincident subruns
    for r in range(len(obs_ids)):
        
        if (len(files_run[0][r]) != len(files_run[1][r])):
            logger.warning(f'For the run {obs_ids[r]}:\nM1 --> {len(files_run[0][r])} subruns')
            logger.warning(f'M2 --> {len(files_run[1][r])} subruns\nKeeping only the coincident ones')
            
            # keeping the coincident subrun number
            if (len(files_run[0][r]) > len(files_run[1][r])):
                files_run[0][r] = files_run[0][r][:len(files_run[1][r])]
                
            else:
                files_run[1][r] = files_run[1][r][:len(files_run[0][r])]
                
    # checking if no files are found for some runs
    empty_runs = 0                                                                                          
    for r in range(len(obs_ids)):
                                                                        
        if (len(files_run[0][r]) == 0 ) and (len(files_run[1][r]) == 0):
            logger.warning(f'No files found for MAGIC run {obs_ids[r]} at {root_MAGIC}')
            empty_runs = empty_runs + 1
            
    # return magic 1 and 2 files
    # if no files found raise an error
    if empty_runs == len(obs_ids):
        logger.warning(f'All MAGIC runs passed are empty, i.e. not files found at: {root_MAGIC}\n\n')

    return files_run[0], files_run[1]


########################################
# ----- find .h5 files for LST-1 ----- #
########################################
# they are located at '/fefs/aswg/data/real/DL1' and are already merged into one file per run
# te output is a list of directories for each input run

def find_LST_dl1(obs_ids, version = 'v0.9'):
    
    '''
    Input:
    --obs_ids : array of run IDs to get the directories
    --version: at the moment the higher version is v0.9, but can be changed to some other
    Output:
    filenames : file directories
    '''
    '''
    Input
    ------------
    --obs_ids: int array
            array of LST run ids
    
    --version: str
            version of the lstchain version used to analyse, currently the last one is v0.9
    Output
    ------------
    --filenames_LST: str array
            filenames for LST-1
           
    '''   

    # --- defining the root paths --- #
    # root folder for all LST subrun-days in the IT cluster
    root_LST = '/fefs/aswg/data/real/DL1'
    
    # in the case the input is not an array (only an integer), we convert it to an array
    logger.debug(f'\nInput ids: {obs_ids}')
    if (type(obs_ids)==int) or (type(obs_ids)==float):
        
        logger.info(f'Input of only one run: {obs_ids}')
        logger.debug(f'Converted {obs_ids} to a list format {[int(obs_ids)]}')
        obs_ids = [int(obs_ids)]
        
    elif (type(obs_ids)!=list) and (type(obs_ids)!=np.ndarray):
        logger.warning(f'\nError: invalid format for "obs_ids", needed an array of runs, or a unique run value')
        sys.exit()
    
    # filenames empty array
    filenames, files_run = [], []

    # --- file tree --- #
    # the folder distribution is:
    # ROOT/date/version/tailcut84/filenameRunXXXXX.XXXX.h5
    
    # iterating over all the dates ------------
    logger.info(f'\nFinding all runs for LST...')
    for date in [name for name in os.listdir(root_LST) if os.path.isdir(os.path.join(root_LST, name))]:
        
        # directory of corresponding date with higher version folder, and entering the "tailcut84" folder
        fnameD = os.path.join(root_LST, date, version, 'tailcut84')

        # checking if for this date-folder exists with subfolders inside
        # in case that don't, we do not search inside for files inside
        if os.path.exists(fnameD):
        
            # now we save the filename of every .h5 merged file ------------
            # iterating over all files inside "tailcut84" folder
            for f in os.listdir(fnameD):

                # first condition holds for only dl1 files
                # second condition holds for not datacheck files
                # third condition restrict to only have subruns files (not the merged runs files)

                if ('.h5' in f) and ('datacheck' not in f) and (f[-8:-7] == '.'):
                    filenames.append(os.path.join(fnameD,f))
                    
        else:
            logger.debug(f'For the date {date}, no folders named {version}/tailcut84 found inside')
    
    # selecting only the coincident runs with the IDs provided ------------     
    logger.info(f'\nSelecting the path to the input obs_ids...')
    for run in obs_ids:
        
        # selecting the subruns, not the merged files in one unique dl1 file i.e. Run{RunN}.h5 format
        coincident_runs = np.array([file for file in filenames if f'Run{run:05}' in file])
        files_run.append(coincident_runs)
        
        logger.info(f'For the LST run {run:05} the number of subruns found are {len(coincident_runs)}')

    # sorting subruns to be in proper order -----------
    for r in range(len(obs_ids)):

        # subruns array
        subruns = []

        # iterating over files-subrun
        for f in files_run[r]:

            # in order to extract the subrun of a file we find the run inside the filename
            ref_index = f.find('Run')
            # and save the subrun number
            subruns.append(int(f[ref_index + 9 : ref_index + 13]))

        # sorted indexes    
        index_mask = np.array(subruns).argsort()
        # sorting the selected files
        files_run[r] = files_run[r][index_mask]        
        

    # return the filenames
    return files_run


# find .root files for MAGIC ####################################################
# the output is a list of directories for each input run
def find_MAGIC_melibea(obs_ids):
    
    '''
    Input:
    --obs_ids : array of run IDs to get the directories
    Output:
    filenames : file directories for the runs
    '''
    '''
    Input
    ------------
    --obs_id: int array
            magic runs array
    
    Output
    ------------
    --filenames: str array
            melibea filenames found for each run
    
    '''    
    
    # --- defining the root paths --- #
    # root directories that we will use
    root_MAGIC_analysed = ['/fefs/aswg/workspace/julian.sitarek/analiza/Crab_2021/an6/data/mars_q/*',
                           '/fefs/aswg/workspace/julian.sitarek/analiza/Crab_2022/mars_q/*']
    root_MAGIC = '/fefs/onsite/common/MAGIC/data/ST/event/Melibea'
  
    logger.info(f'Main melibea files root folder is {root_MAGIC}')
    logger.info(f'And other apart analysed runs in:')
    for root in root_MAGIC_analysed:
        logger.info(f'--> {root}')
    
    # searching analyised runs in the analysed directories
    filenames_analysed = [glob.glob(root_MAGIC) for root_MAGIC in root_MAGIC_analysed]
    filenames_analysed = np.concatenate(filenames_analysed)
    filenames_analysed.sort()
    logger.info(f'\nFound {len(filenames_analysed)} files already analysed')
    
    # taking the run numbers
    runs_analysed = []
    for f in filenames_analysed:
        index = f.find('/mars_q/')
        runs_analysed.append(int(f[index+8+9:index+8+9+8]))
    logger.debug(f'For the runs: {runs_analysed}')

    # both files for M1 and M2
    filenames = [] # all filenames
    files_run = [] # filtered filenames
    
    # --- file tree --- #
    # the directories have the following structure
    # 'root/year/month/day/***.root'
    logger.info(f'\nFinding all runs...')
    
    # iterating over years -----------
    for year in [name for name in os.listdir(root_MAGIC) if os.path.isdir(os.path.join(root_MAGIC, name))]:
        # year filename root/year
        fnameY = os.path.join(root_MAGIC, year)

        # iterating over months -----------
        for month in [name for name in os.listdir(fnameY) if os.path.isdir(os.path.join(fnameY, name))]:
            # month filename root/year/month
            fnameM = os.path.join(fnameY, month)

            # iterating over days -----------
            for day in [name for name in os.listdir(fnameM) if os.path.isdir(os.path.join(fnameM, name))]:
                # day filename root/year/month/day
                fnameD = os.path.join(fnameM, day)

                # iterating over files in main folder -----------
                for f in os.listdir(fnameD):

                    # avoiding signal and other filenames
                    if ('melibea' not in f) and ('.root' in f):

                        # appending every filename
                        filenames.append(os.path.join(fnameD,f))

        
    # selecting only the coincident runs -----------
    logger.info(f'\nSelecting the paths to the input obs_ids...\n')
    for run in obs_ids:
        
        # first of all we search if the run has been analysed
        if run in runs_analysed:
                  
            index_run = runs_analysed.index(run)
            files_run.append(filenames_analysed[index_run])
            
            logger.debug(f'For the MAGIC run {run:08} analysed-file is found')        
        
        
        else:
        
            coincident_runs = np.array([file for file in filenames if f'_{run:08}_' in file])

            if len(coincident_runs) == 0:
                logger.warning(f'For the MAGIC run {run:08} no files found')

            else:
                files_run.append(coincident_runs[0])

                logger.debug(f'For the MAGIC run {run:08} {len(coincident_runs)} file is found')

    # return filenames
    return files_run

