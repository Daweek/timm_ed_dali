#!/bin/sh
#$ -l rt_F=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/$JOB_ID_render
#$ -cwd
#$ -N fdb_render

# cat $JOB_SCRIPT
echo "................................................................................"
echo "JOB ID: ---- >>>>>>   $JOB_ID"

# ======== Modules ========
source /etc/profile.d/modules.sh
module purge
module load cuda/12.4/12.4.0 cudnn/9.1/9.1.1 nccl/2.21/2.21.5-1 gcc/13.2.0 cmake/3.29.0 hpcx-mt/2.12

# ======== Pyenv/ ========
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
# size=(32)

# NFS network_________________
#root=nfs/raw

# SSD ___________________________
export SSD=/local/${JOB_ID}.1.gpu
root=${SSD}/raw
# Check if the directory exists
if [ -d "$root" ]; then
    echo "Directory '$root' already exists."
else
    echo "Directory '$root' does not exist. Creating it now..."
    mkdir "$root"
fi
#SSD___________________________________


for imsize in "${size[@]}"
do
    echo "______Start rendering_________"

    start_time=$(date +%s%3N)

    # Render to somewhere
    # mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -np 80 python mpi_gpu.py --ngpus-pernode 4 --image_res $imsize --save_root ${root}/fdb1k_${imsize}x 
    mpirun --bind-to socket --use-hwthread-cpus -np 80 python mpi_gpu.py --ngpus-pernode 4 --image_res $imsize --save_root ${root}/fdb1k_${imsize}x
    
    #--instance 10 --rotation 1 --nweights 1

    end_time=$(date +%s%3N)
    total_duration=$((end_time - start_time))
    times+=("$total_duration")

    du -sh ${root}/fdb1k_${imsize}x_egl

    echo "Number of images generated:"
    find ${root}/fdb1k_${imsize}x_egl -type f -print |wc -l | xargs printf "%'d\n"
    
    echo "This experiment Duration: ${imsize}x${imsize} "
    convert_milliseconds "$total_duration"
    echo "______Finish_________"
    echo "                   "
done

echo "                   "
echo "                   "

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