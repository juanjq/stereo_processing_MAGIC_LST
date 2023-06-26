#! /bin/bash

###########################
# select the source 
source_name="Mrk421"
###########################

file="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/config_files/common_data$source_name.txt"

# read the lines of the given file
while read -r line; do

# only operating if the line is not commented with #
if  [[ "${line:0:1}" != '#' ]]; then
run_str="$line"

echo "#! /bin/bash" > stereo_to_dl2_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_stereo_to_dl2/script_stereo_to_dl2.py stereo_dl1 '$run_str' '$source_name'" >> stereo_to_dl2_tmpjob.sh

echo -e "Sending MAGIC Run $run_str calib --> dl1 job (using short queue always)"
sbatch -p short --mem=50000 --output="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_stereo_to_dl2/output_slurm/slurm-%j.out" stereo_to_dl2_tmpjob.sh

fi

done < $file
