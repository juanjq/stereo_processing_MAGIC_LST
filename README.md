# stereo_processing_magic_lst
## Pipeline Description

A detailed description of all the steps in the pipeline are described below. As well as step-by-step instructions on how to use it. The scripts are prepared to run in the IT cluster, since it uses files that are there. But you can still run it from outside if you download the files and use them from another location, changing the paths. All the scripts and notebooks can be downloaded from this GitHub that as been prepared for this work [[1]](https://github.com/juanjq/stereo_processing_magic_lst/blob/main/thesis_stereo_magic%26lst1.pdf).

### Data Levels and File Locations

As explained in [Section 1](#data-levels) and depicted in [Figure 1](#flowchart), the already available data is, for the case of MAGIC; the calibrated data, and in the case of LST-1; the DL1 stage. MAGIC data is found at `/fefs/onsite/common/MAGIC/data/MX/event/Calibrated` (with `MX` being `M1` for MAGIC-1 and `M2` for MAGIC-2) and LST-1 DL1 data at `/fefs/aswg/data/real/DL1`. Also the MC files are already processed, and are located at the respective subdirectory of `/fefs/aswg/LST1MAGIC/mc/models`.

### Configuration and Parameter Selection

Before starting with the running of the pipeline, some considerations should be taken and the files need to be configured for the analysis you want to perform. The main parameters that can be changed in the `config.yaml` file inside the `config_files` folder. The parameters that can be selected are summarized here:

- **Source name:** The source that is being studied can be changed. Using the `source_name` variable in `config.yaml` the source can be specified by its standard name. Otherwise the exact coordinates can also be specified.
- **Interpolation method:** Changing `interpolation_method` in `config.yaml` to `nearest` (no interpolation), `linear` or `cubic` the interpolation method to use the IRFs can be specified.
- **Cut types:** For the cuts applied in gammaness and θ² a fixed cut or a dynamic cut can be specified with the variable `cut_type` in `config.yaml`. For a fixed cut, use `global`, and for a dynamic one `dynamic`. The cut values or efficiencies can also be specified.
- **Mean method:** The method to apply the mean of parameters can be specified in the scripts `script_merge_and_mean.py` in the `bash_scripts/bash_merge_and_mean` folders. As explained in [Section 2](#implementation), the weight types can be selected changing the variable `weight_type` to `simple`, `intensity` or `variance`.
- **MC files:** The MC directories should be changed in order to analyse other sources different than the Crab Nebula. The correct declination and NSB should be selected.
- **Selecting runs:** In order to select the runs you want to analyse of a determined source, the notebook `create_configuration_txt_files.ipynb` inside the folder `notebooks_data_generation` will be used to read the stereo log archive (the updated version can be downloaded from [here](https://docs.google.com/spreadsheets/u/1/d/1Tya0tlK-3fuN6_vXOU5FwJruis5vBA9FALAPLFHOfBQ/edit)). In this notebook the runs can be filtered by source name, amount of observation time of observations, type of wobble, time of same wobble, and many other parameters. Once filtered, two configuration `.txt` files will
