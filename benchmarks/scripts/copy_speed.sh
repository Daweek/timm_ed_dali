#!/bin/sh
#$ -l rt_F=128
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/$JOB_ID_copy_parallel
#$ -cwd
#$ -N speed_copy

cat $JOB_SCRIPT
echo "................................................................................"
echo "JOB ID: ---- >>>>>>   $JOB_ID"

# ======== Modules and Python on main .configure.sh ==================================

source ../config.sh

######################################################################################

# Main Varibales of speed test
global_start=$(date +%s%3N)
# Experiments parameters
size=(32 64 128 256 512 1024)
# size=(32)

export SSD=/local/${JOB_ID}.1.gpu

echo "Start copying experiment using parallel command"
date +"%Y-%m-%d-%H:%M:%S"


for imsize in "${size[@]}"
do
    echo "______Start Copying_________"
    start_time=$(date +%s%3N)

    parallel -j 80 --eta rsync -a {} ${SSD}/fdb1k_${imsize}x_egl ::: nfs/raw/fdb1k_${imsize}x_egl/*
    # parallel -j 80 --eta cp -r {} ${SSD}/fdb1k_${imsize}x_egl ::: nfs/raw/fdb1k_${imsize}x_egl/*

    end_time=$(date +%s%3N)
    total_duration=$((end_time - start_time))
    times+=("$total_duration")

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