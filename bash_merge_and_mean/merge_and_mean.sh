#! /bin/bash

###########################
# select the source 
source_name="BLLac"
###########################

echo "#! /bin/bash" > merge_and_mean_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_merge_and_mean/script_merge_and_mean.py process '$source_name' '0'" >> merge_and_mean_tmpjob.sh

echo -e "\nSending LST1 + MI + MII merging and mean"
sbatch -p long --mem=20000 --output="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_merge_and_mean/output_slurm/slurm-%j.out" merge_and_mean_tmpjob.sh

echo "#! /bin/bash" > merge_and_mean_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_merge_and_mean/script_merge_and_mean.py process '$source_name' '1'" >> merge_and_mean_tmpjob.sh

echo -e "\nSending MI + MII merging and mean"
sbatch -p long --mem=20000 --output="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_merge_and_mean/output_slurm/slurm-%j.out" merge_and_mean_tmpjob.sh



