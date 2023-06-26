#! /bin/bash

###########################
# select the source 
source_name="BLLac"
###########################


# --- common data file --- # 
file="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/config_files/MAGIC_runs$source_name.txt"

# read the lines of the given file
while read -r line; do
# and only operating if the line is not commented with #
if  [[ "${line:0:1}" != '#' ]]; then
run_str="$line"

# each line will be different MAGIC runs, so we iterate for each run
for run in ${run_str//,/ }; do 

# checking first if the MAGIC run have files available
python /fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_magic_cal_to_dl1/script_magic_cal_to_dl1.py check_sruns $run; nsruns=$?

if [[ "$nsruns" -ne "0" ]]; then

echo "#! /bin/bash" > cal_to_dl1_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_magic_cal_to_dl1/script_magic_cal_to_dl1.py cal_dl1 '$run' '$source_name'" >> cal_to_dl1_tmpjob.sh

# sending to long or short queue if needed (around 25 subruns is the limit)
if [[ "$nsruns" -lt "25" ]]; then
echo -e "Sending MAGIC Run $run calib --> dl1 job (using short queue because $nsruns subruns found < 25)"
sbatch -p short --mem=13000 --output="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_magic_cal_to_dl1/output_slurm/slurm-%j.out" cal_to_dl1_tmpjob.sh

# if number of subruns higher than 25, we send the job to long queue
else
echo -e "Sending MAGIC Run $run calib --> dl1 job (using long queue because $nsruns subruns found >= 25)"
sbatch -p long --mem=17000 --output="/fefs/aswg/workspace/juan.jimenez/stereo_analysis/bash_magic_cal_to_dl1/output_slurm/slurm-%j.out" cal_to_dl1_tmpjob.sh

fi
echo ""

else
echo -e "No files found for the MAGIC run $run \nPassing to next run\n"

fi
done
fi

done < $file
