#!/bin/sh
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -o output/$JOB_ID_ffcv
#$ -cwd
#$ -N ffcv

cat $JOB_SCRIPT
echo "....................................................................................."
echo "JOB ID: ---- >>>>>>   $JOB_ID"

# ======== Modules ========
source /etc/profile.d/modules.sh
module purge
module load cuda/12.4/12.4.0 cudnn/9.1/9.1.1 nccl/2.21/2.21.5-1 gcc/13.2.0 cmake/3.29.0 hpcx-mt/2.12

======== Pyenv/ ========
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
pyenv local anaconda3-2023.07-2/envs/ffcv

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

convert_milliseconds() {
    local total_ms=$1

    # Calculate total seconds and remaining milliseconds
    local total_seconds=$(echo "$total_ms / 1000" | bc)
    local remaining_ms=$((total_ms % 1000))  # Remaining milliseconds

    # Calculate days, hours, minutes, and seconds using bc
    local days=$(echo "$total_seconds / 86400" | bc)
    local hours=$(echo "($total_seconds % 86400) / 3600" | bc)
    local minutes=$(echo "($total_seconds % 3600) / 60" | bc)
    local seconds=$(echo "$total_seconds % 60" | bc)

    # Print in D:H:M:S.ms format
    printf "\t%d days, %02d hours, %02d minutes, %02d seconds, %03d milliseconds\n" "$days" "$hours" "$minutes" "$seconds" "$remaining_ms"
}


#######################################################
global_start=$(date +%s%3N)
# Experiments parameters
size=(32 64 128 256 512 1024)
# size=(32 64)

# IN->>>>>SSD ___________________________
export SSD=/local/${JOB_ID}.1.gpu
in_root=${SSD}/raw
#SSD___________________________________

# OUT->>>>>SSD ___________________________
export SSD=/local/${JOB_ID}.1.gpu
out_root=${SSD}/ffcv
# Check if the directory exists
if [ -d "$out_root" ]; then
    echo "Out Directory '$out_root' already exists."
else
    echo "Out Directory '$out_root' does not exist. Creating it now..."
    mkdir "$out_root"
fi
#SSD___________________________________

# NFS network_________________
# in_root=nfs/raw
# out_root=nfs/ffcv

export MAX_RES=362
# Compress probaility = 0.25 -> # Compress a random 1/4 of the dataset.
export COMP_PROB=0.10 
export QUALITY=90

for imsize in "${size[@]}"
do
    echo "______Start Betoning_________"

    start_time=$(date +%s%3N)

    # Render to somewhere
    python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${in_root}/fdb1k_${imsize}x_egl --cfg.write_path=${out_root}/fdb1k_${imsize}x_${QUALITY}jpeg_qualiy.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}

    end_time=$(date +%s%3N)
    total_duration=$((end_time - start_time))
    times+=("$total_duration")

    du -sh ${out_root}/fdb1k_${imsize}x_${QUALITY}jpeg_qualiy.ffcv
    echo "______Finish_________"
    echo "                   "
done

for ((i=0; i<${#size[@]}; i++)); do

    echo "FINAL results from resolution ${size[$i]} x ${size[$i]}:"
    convert_milliseconds "${times[$i]}"

done
echo "                   "
echo "                   "

global_end=$(date +%s%3N)
global_duration=$((global_end - global_start))
echo "TOTAL TIME OF THE EXPERMIENT: "
convert_milliseconds "$global_duration"

echo "                   "
echo "                   "
echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#