# stereo_processing_magic_lst
## Pipeline Description

A detailed description of all the steps in the pipeline are described below. As well as step-by-step instructions on how to use it. The scripts are prepared to run in the IT cluster, since it uses files that are there. But you can still run it from outside if you download the files and use them from another location, changing the paths. All the scripts and notebooks can be downloaded from this GitHub that as been prepared for this work [[1]](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf).

  ![alt text](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/images/stereo-processing-flowchart-1.png)

### Data Levels and File Locations

As explained in [Section 1](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf), the already available data is, for the case of MAGIC; the calibrated data, and in the case of LST-1; the DL1 stage. MAGIC data is found at `/fefs/onsite/common/MAGIC/data/MX/event/Calibrated` (with `MX` being `M1` for MAGIC-1 and `M2` for MAGIC-2) and LST-1 DL1 data at `/fefs/aswg/data/real/DL1`. Also the MC files are already processed, and are located at the respective subdirectory of `/fefs/aswg/LST1MAGIC/mc/models`.

### Configuration and Parameter Selection

Before starting with the running of the pipeline, some considerations should be taken and the files need to be configured for the analysis you want to perform. The main parameters that can be changed in the `config.yaml` file inside the `config_files` folder. The parameters that can be selected are summarized here:

- **Source name:** The source that is being studied can be changed. Using the `source_name` variable in `config.yaml` the source can be specified by its standard name. Otherwise the exact coordinates can also be specified.
- **Interpolation method:** Changing `interpolation_method` in `config.yaml` to `nearest` (no interpolation), `linear` or `cubic` the interpolation method to use the IRFs can be specified.
- **Cut types:** For the cuts applied in gammaness and $\theta^2$ a fixed cut or a dynamic cut can be specified with the variable `cut_type` in `config.yaml`. For a fixed cut, use `global`, and for a dynamic one `dynamic`. The cut values or efficiencies can also be specified.
- **Mean method:** The method to apply the mean of parameters can be specified in the scripts `script_merge_and_mean.py` in the `bash_scripts/bash_merge_and_mean` folders. As explained in [Section 2](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf), the weight types can be selected changing the variable `weight_type` to `simple`, `intensity` or `variance`.
- **MC files:** The MC directories should be changed in order to analyse other sources different than the Crab Nebula. The correct declination and NSB should be selected.
- **Selecting runs:** In order to select the runs you want to analyse of a determined source, the notebook `create_configuration_txt_files.ipynb` inside the folder `notebooks_data_generation` will be used to read the stereo log archive (the updated version can be downloaded from [here](https://docs.google.com/spreadsheets/u/1/d/1Tya0tlK-3fuN6_vXOU5FwJruis5vBA9FALAPLFHOfBQ/edit)). In this notebook the runs can be filtered by source name, amount of observation time of observations, type of wobble, time of same wobble, and many other parameters. Once filtered, two configuration `.txt` files will be produced.
  
  ![alt text](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/images/example_table_stereo_data.png)

### Running the Pipeline

Once the configuration files have been prepared, we can start running the scripts to obtain the processed files and then the physical results. The notebooks need to be run with the Anaconda environment `magic-cta-pipe` [\[2\]](https://github.com/cta-observatory/magic-cta-pipe). The steps that need to be performed, in the order that they should be applied, are the following:

1. **MAGIC calibration to DL1:** The DL1 data is already computed for LST-1, but this is not the case for MAGIC. So the first step will be to process the MAGIC calibrated data to DL1. The starting point will be subrun-wise calibrated MAGIC files, a set of files per run and per telescope. In the prepared GitHub, inside the folders `bash_scripts/bash_magic_cal_to_dl1`, the Bash script `cal_to_dl1.sh` sends the fragmented jobs to the IT cluster queues. This script will generate a folder per observation night (a group of 2 to 10 runs), containing the DL1 MAGIC files merged run-wise.

2. **DL1 stereo coincidences:** Once we have all the data in the same stage and format (DL1), the stereo information can be extracted by finding the coincidence of events. The Bash script `dl1_to_stereo.sh` located in the folders `bash_scripts/bash_dl1_to_stereo` needs to be run. This script finds the coincident events between telescopes, as explained in [Section 3](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf), and creates a unique DL1 file per telescope. The files are stored in a folder for each LST-1 run in subrun-wise format.

3. **DL1 to DL2:** All DL1 data is processed with the scripts to DL2. In this step, the RF files need to be input, and the source path can be changed to use different RFs. The script `stereo_to_dl2.sh` is located in the GitHub folder `bash_scripts/bash_stereo_to_dl2`. Running it will generate a DL2 file per run and store it in a specified unique folder.

4. **Create IRFs and DL2 to DL3:** Before obtaining the DL3 data, the specific IRFs need to be produced. The notebook `create_irfs_and_dl2_to_dl3.ipynb` inside the folder `notebooks_dl2_to_dl3` in the GitHub repository should be run. The first part of this notebook generates the IRFs given the required MC files and stores them in one folder. Then, the second part of the notebook processes the DL2 data to DL3, creating the respective index files. The DL3 data is also stored run-wise in a unique folder.

5. **Standalone datasets:** In order to compare the data obtained with this analysis with data obtained with individual telescopes' analysis, the notebook `create_lst_only_hdf.ipynb` inside the `notebooks_data_generation` folder should be run to obtain the LST-1 standalone dataset. For the case of MAGIC, the notebook `create_melibea_hdf.ipynb` in the same folder should be run. Both notebooks will select the exact same events that have been analyzed for the three telescopes and create a DL2 dataset with these events in `.hdf5` format.

6. **MC dataset:** To make comparisons with gamma-ray MCs, a dataset with enough statistics of MC simulations can be created using the notebook `create_mc_dataframes.ipynb` inside the `notebooks_data_generation` folder. A unique `.hdf` file of DL2 data will be created.

7. **Mean and merge DL2:** This set of scripts uses the tool `get_dl2_mean` from `magic-cta-pipe` to process the DL2 data obtained for each telescope and perform a mean of the parameters (weighted by the inverse of the variances). Since the direction, energy, and gammaness parameters are calculated for each telescope, a mean should be computed to obtain a unique value for a given event (as explained in [Section 4](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf)). The bash script `merge_and_mean.sh` inside the `bash_scripts/bash_merge_and_mean` folder should be run (it can be sent to the queue of the IT cluster). The script also generates a DL2 merged dataset with all the runs that have been processed.

8. **Coincidences datasets:** In the second step of this chain, the event coincidence finding has been performed. In the generated files, there is more information than the DL1 data. Additional information is also stored in the keys `coincidence/profile` and `coincidence/feature` of the `.hdf5` files. The former contains the number of coincident events for the different tried time offsets (see [Section 3](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf)), and the latter contains the timestamps and pointings of the different telescopes for each event. This information is merged for all runs and stored in two unique `.hdf5` files. This is achieved by running the notebook `create_coincidences_datasets.ipynb` inside the `notebooks_data_generation` folder.

9. **Add complementary data:** The last step to perform is adding some additional data to the DL2 files. These scripts add data to the previously generated files with the selected number of off regions. It calculates the $\theta^2$ parameter (see [Section 3](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf)) for the ON and OFF regions. It also adds other supplementary information, such as the zenith distance (from the altitude coordinate), for convenience in further analysis. The scripts can be run with the notebook `dl2_additional_data.ipynb` inside the `notebooks_data_generation` folder.

With that, all the data has been processed, and different files with all the information have been stored. Now the physical analysis can start to be done. It can be divided into three main parts:

1. **Data sample analysis:** The notebooks used to analyze the dataset itself, in order to characterize it well. Also, the quality of the data is checked and runs can be filtered. The notebooks are located in the `notebooks_analysis_dataset` folder.

2. **DL2 analysis:** The notebooks located in the `notebooks_analysis_dl2` folder are generally used to analyze different aspects of the DL2 data. The gammaness, intensity, energy, IRFs, etc., are analyzed in the different notebooks. In the `arrival_positions` sub-folder, the artifact explained in [Section 5](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf) is analyzed in different notebooks.

3. **DL3 analysis:** The higher-level analysis is done using Gammapy, so the Anaconda environment that should be used now needs to contain the latest version of Gammapy. The `gammapy-v1.0` environment can be used.
