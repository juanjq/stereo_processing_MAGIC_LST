import numpy  as np 
import pandas as pd
import os, sys, glob

sys.path.insert(0, '/fefs/aswg/workspace/juan.jimenez/stereo_analysis/scripts')
import auxiliar as aux
import merge

# --- logging --- #
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

from magicctapipe.io import get_dl2_mean

def process(source_name, step, merge_runs=True, combo_types=True):
    
    step = int(step)
    logger.info(f'Processing {source_name} data...')
    
    #############################
    # ------- parameters ------ #
    #############################
    # simple, variance or intesity
    weight_type = 'variance'

    # --- directories --- #
    common_data_file = f'/fefs/aswg/workspace/juan.jimenez/stereo_analysis/config_files/common_data{source_name}.txt'
    main_dir         = '/fefs/aswg/workspace/juan.jimenez/data/dl2'
    stereo_dl2_dir        = os.path.join(main_dir, f'stereo_raw_dl2_{source_name}')

    stereo_dl2_merged_dir = os.path.join(main_dir, f'stereo_merged_{source_name}')
    stereo_dl2_mean_dir   = os.path.join(main_dir, f'stereo_mean')
    #############################
    
    # creating folders
    aux.createdir(stereo_dl2_merged_dir)
    aux.createdir(stereo_dl2_mean_dir)

    ########################################
    # --- converting to totaldataframe --- #
    ########################################
    # reading dataframes and putting all together into a unique dataframe
    # --- converting to unique dataframe --- #
    if merge_runs:
        merge.stereo_dl2(input_dir=stereo_dl2_dir, output_dir=stereo_dl2_merged_dir)

    # now we read the file
    # first of all we find the file
    merged_files = glob.glob(os.path.join(stereo_dl2_merged_dir,'*'))
    merged_file  = [f for f in merged_files if '_to_' in f][0]

    # reading the dataframe
    df_total = pd.read_hdf(merged_file, key='events/parameters')
    df_total.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    df_total.sort_index(inplace=True)
    ######################################## 
    
    ####################################
    # --- filtering by combo types --- #
    ####################################
    if combo_types:

        # calculating merged dataframes with combo types
        df_merged = [df_total.query(f'combo_type == {ctype}', inplace=False) for ctype in [3, 0, 1, 2]]

        # number of events and statistics
        nevents3, nevents0 = int(len(df_merged[0])/3), int(len(df_merged[1])/2)
        nevents1, nevents2 = int(len(df_merged[2])/2), int(len(df_merged[3])/2)

        neventstotal = nevents3 + nevents0 + nevents1 + nevents2 
        print(f'\n3-Telescopes events filtering\nPassing from {neventstotal} events to...')
        print(f'LST1 + MI + MII --> to {nevents3} events,\t{nevents3 / neventstotal * 100:.2f}% conserved')
        print(f'MI + MII        --> to {nevents0} events,\t{nevents0 / neventstotal * 100:.2f}% conserved')
        print(f'LST1 + MI       --> to {nevents1} events,\t{nevents1 / neventstotal * 100:.2f}% conserved')
        print(f'LST1 + MII      --> to {nevents2} events,\t{nevents2 / neventstotal * 100:.2f}% conserved')


        for name, i in zip(['3tel', 'MI_MII', 'LST1_MI', 'LST1_MII', 'all_combo'], range(len(df_merged)+1)):

            if i != len(df_merged):
                df_tmp = df_merged[i]
            else:
                df_tmp = df_total
                
            # --- creating a unique identification --- #
            print(f'\nCreating a unique \'run.event\' identification label for {name}')
            # extracting indexes, runs and events
            obs_id_array   = np.array(df_tmp.index.get_level_values('obs_id').values).astype(str)
            event_id_array = np.array(df_tmp.index.get_level_values('event_id').values).astype(str)
            universal_id = np.char.add(obs_id_array, np.char.add('.',event_id_array))

            obs_id_array_magic   = df_tmp['obs_id_magic'].to_numpy().astype(str)
            event_id_array_magic = df_tmp['event_id_magic'].to_numpy().astype(str)
            universal_id_magic = np.char.add(obs_id_array_magic, np.char.add('.',event_id_array_magic))

            df_tmp.loc[:, 'total_id'] = universal_id
            df_tmp.loc[:, 'magic_id'] = universal_id_magic  

            df_tmp.to_hdf(f'{stereo_dl2_merged_dir}/dl2_merged_{source_name}_total.{name}.h5', key='events/parameters')

    
    dfs = [df_total, df_total.query(f'combo_type == {3}', inplace=False)]
    name = ['all_combo', '3tel']
                     
    ##############################
    # --- for all telescopes --- #
    ##############################
    # calculating means
    print(f'Calculating means for all telescopes and {name[step]} events')      
    df_mean = get_dl2_mean(dfs[step], weight_type=weight_type)

    # --- creating a unique identification --- #               
    obs_id_array   = np.array(df_mean.index.get_level_values('obs_id').values).astype(str)
    event_id_array = np.array(df_mean.index.get_level_values('event_id').values).astype(str)
    universal_id = np.char.add(obs_id_array, np.char.add('.',event_id_array))
    df_mean.loc[:, 'total_id'] = universal_id

    # --- create .h5 file --- #
    print('Creating .h5 file\n')
    df_mean.to_hdf(f'{stereo_dl2_mean_dir}/dl2_mean_{source_name}_total.{name[step]}.h5', key='events/parameters')
    ##############################                      

                           
                           
    ##############################
    # -------- for LST --------- #
    ##############################
    # calculating means
    print(f'Calculating means for LST and {name[step]} events')  
    df_tmp_LST = dfs[step].query('tel_id == 1', inplace=False)
    df_mean = get_dl2_mean(df_tmp_LST, weight_type=weight_type)

    # --- creating a unique identification --- #               
    obs_id_array   = np.array(df_mean.index.get_level_values('obs_id').values).astype(str)
    event_id_array = np.array(df_mean.index.get_level_values('event_id').values).astype(str)
    universal_id = np.char.add(obs_id_array, np.char.add('.',event_id_array))
    df_mean.loc[:, 'total_id'] = universal_id

    # --- create .h5 file --- #
    print('Creating .h5 file\n')
    df_mean.to_hdf(f'{stereo_dl2_mean_dir}/dl2_mean_{source_name}_LST.{name[step]}.h5', key='events/parameters')
    df_tmp_LST.to_hdf(f'{stereo_dl2_merged_dir}/dl2_merged_{source_name}_LST.{name[step]}.h5', key='events/parameters')
    ##############################


    ##############################
    # ------- for MAGIC -------- #
    ##############################
    # calculating means
    print(f'Calculating means for MAGIC and {name[step]} events')  
    df_tmp_MAGIC = dfs[step].query('tel_id == 2 | tel_id == 3', inplace=False)
    df_mean = get_dl2_mean(df_tmp_MAGIC, weight_type=weight_type)

    # --- creating a unique identification --- #               
    obs_id_array   = np.array(df_mean.index.get_level_values('obs_id').values).astype(str)
    event_id_array = np.array(df_mean.index.get_level_values('event_id').values).astype(str)
    universal_id = np.char.add(obs_id_array, np.char.add('.',event_id_array))
    df_mean.loc[:, 'total_id'] = universal_id

    # --- create .h5 file --- #
    print('Creating .h5 file\n')
    df_mean.to_hdf(f'{stereo_dl2_mean_dir}/dl2_mean_{source_name}_MAGIC.{name[step]}.h5', key='events/parameters')
    df_tmp_MAGIC.to_hdf(f'{stereo_dl2_merged_dir}/dl2_merged_{source_name}_MAGIC.{name[step]}.h5', key='events/parameters')
    ##############################           

if __name__ == '__main__': 
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3])
