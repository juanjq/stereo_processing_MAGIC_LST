#! /bin/bash

###########################
# select the source 
source_name="Mrk421"
###########################

file="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/config_files/common_data$source_name.txt"

# first of all ordering the folders in the night dl1 directory
echo "#! /bin/bash" > dl1_to_stereo_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_dl1_to_stereo/script_order_night_folders.py '$source_name'" >> dl1_to_stereo_tmpjob.sh
echo -e "Ordering night folders:"
sbatch -p short --output="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_dl1_to_stereo/output_slurm/slurm-%j.out" dl1_to_stereo_tmpjob.sh

# read the lines of the given file
while read -r line; do

# only operating if the line is not commented with #
if  [[ "${line:0:1}" != '#' ]]; then
run_str="$line"

echo "#! /bin/bash" > dl1_to_stereo_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_dl1_to_stereo/script_dl1_to_stereo.py stereo_dl1 '$run_str' '$source_name'" >> dl1_to_stereo_tmpjob.sh

echo -e "Sending MAGIC Run $run_str calib --> dl1 job (using long queue always)"
sbatch -p long --mem=7000 --output="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_dl1_to_stereo/output_slurm/slurm-%j.out" dl1_to_stereo_tmpjob.sh

fi

done < $file
