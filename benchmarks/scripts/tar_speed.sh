#!/bin/sh
#$ -l rt_F=64
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/$JOB_ID_tar_parallel
#$ -cwd
#$ -N tar_speed

cat $JOB_SCRIPT
echo "................................................................................"
echo "JOB ID: ---- >>>>>>   $JOB_ID"

# ======== Modules and Python on main .configure.sh ==================================

source ../config.sh

######################################################################################
# Main Varibales of speed test
export WORKERS=80

global_start=$(date +%s%3N)
# Experiments parameters
size=(32 64 128 256 512 1024)
#size=(32)
dirs=$(seq -w 00000 00999)

export SSD=/local/${JOB_ID}.1.gpu

export LOCATION=/groups/3/gcc50533/acc12930pb/datasets/nfs/tars
#export LOCATION=${SSD}


echo "Start copying experiment using parallel command"
date +"%Y-%m-%d-%H:%M:%S"


for imsize in "${size[@]}"
do
    echo "______Start Taring_________"
    start_time=$(date +%s%3N)

    # Check if the directory exists for nfs
    tar_dir=${LOCATION}/${JOB_ID}_${HOSTNAME}_fdb1k_${imsize}x_egl
    # tar_dir=${LOCATION}/fdb1k_${imsize}x_egl
    if [ -d "$tar_dir" ]; then
        echo "Directory '$tar_dir' already exists."
    else
        echo "Directory '$tar_dir' does not exist. Creating it now..."
        mkdir "$tar_dir"
    fi

    # Tar using the parallel engine....
    echo "Changing directory to inlcude only dirs.. into tars..."
    cd nfs/raw/fdb1k_${imsize}x_egl
    pwd -LP
    
    parallel -j $WORKERS --eta tar -cf ${tar_dir}/{}.tar {} ::: ${dirs}

    echo "Returning to parent directory../"
    cd ../../../
    pwd -LP

    # Untar directly to SSD...
     echo "Star UNTARING.... ----------<<<<<<<<<<<<<<<"
    # Check if the directory exists for ssd
    untar_dir=${SSD}/fdb1k_${imsize}x_egl
    if [ -d "$untar_dir" ]; then
        echo "Directory '$untar_dir' already exists."
    else
        echo "Directory '$untar_dir' does not exist. Creating it now..."
        mkdir "$untar_dir"
    fi

    parallel -j $WORKERS --eta tar -xf ${tar_dir}/{}.tar -C ${untar_dir}/ ::: ${dirs}

    echo "Number of images Untared:"
    find ${untar_dir} -type f -print |wc -l | xargs printf "%'d\n"

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