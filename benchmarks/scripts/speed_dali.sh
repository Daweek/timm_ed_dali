#!/bin/sh
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/$JOB_ID_speed_dali
#$ -cwd
#$ -N speed10ep

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
#size=(32 64)

export SSD=/local/${JOB_ID}.1.gpu

# NFS network_________________
# export SSD=nfs/raw

# Checking arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 arg1 arg2"
    exit 1
fi
arg1=$1
arg2=$2

if [ "$arg2" == "nfs" ]; then
    echo "Select files from NFS"
    export LOCATION=nfs
elif [ "$arg2" == "ssd" ]; then
echo "Select files from SSD"
    export LOCATION=${SSD}
    ########################### COPY files
    # echo "Copying RAW files to SSD"
    # cp -rfv nfs/raw ${SGE_LOCALDIR}
    # echo "Finish copying files..."

    ########################### RENDER files
    echo "Render... RAW files to SSD"
    cd ../render_engines/fdb
    sh ./scripts/render_ssd.sh
    cd ../../benchmarks
    echo " "
    pwd -LP
    echo " "
    echo " "
    echo "Continue with speed test..."

else
    echo "Unknown command: $arg2"
    exit
fi

# Optionally perform actions based on the arguments
if [ "$arg1" == "p" ]; then
    echo "Select Pytorch PIPELINE."
    export TYPE=${LOCATION}/raw
    export PIPE='PyTo'
    export PIPE_FLAGS=' '
elif [ "$arg1" == "d" ]; then
    echo "Select Dali PIPELINE."
    export TYPE=${LOCATION}/raw
    export PIPE='DaLi'
    export PIPE_FLAGS='--dali'
elif [ "$arg1" == "f" ]; then
    echo "Select Fccv PIPELINE."
    export TYPE=${LOCATION}/ffcv
    export PIPE='FFcv'
    export PIPE_FLAGS='-f'
    # /local/43134625.1.gpu/ffcv/fdb1k_32x_90jpeg_qualiy.ffcv -f
elif [ "$arg1" == "w" ]; then
    echo "Select Wds PIPELINE."
    export TYPE=${LOCATION}/wds
    export PIPE='Wds'
    export PIPE_FLAGS='-w --wds-datasetlen 1000000'
    # --data-dir "/local/43134624.1.gpu/wds/fdb1k_32x_egl/fdb1k_32x_egl-train-{000000..000099}.tar" -w --wds-datasetlen 1000000 
else
    echo "Unknown command: $arg1"
    exit
fi



# SSD ___________________________
checkpoint=${SSD}/checkpoint
# Check if the directory exists
if [ -d "$checkpoint" ]; then
    echo "Directory '$checkpoint' already exists."
else
    echo "Directory '$checkpoint' does not exist. Creating it now..."
    mkdir "$checkpoint"
fi
#SSD___________________________________

## Training variables ####################################################
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=2042
export NGPUS=4
export NUM_PROC=4
# export PIPE=PyTo
export STORAGE=ssd

export MODEL=tiny
export LR=1.0e-3
export CLS=1
export EPOCHS=1
export LOCAL_BATCH_SIZE=64
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224

export OUT_DIR=${SSD}/checkpoint/${MODEL}/fdb${CLS}k/pre_training
wandb disabled
#######################################################################

for imsize in "${size[@]}"
do
    echo "______Start Training_________"
    start_time=$(date +%s%3N)

    export SSD=/local/${JOB_ID}.1.gpu
    export RENDER_HWD=files
    export EXPERIMENT=SpeedTest_${imsize}x_$(date +"%Y-%m-%d-%H:%M:%S")

    if [ "$arg1" == "p" ]; then
        export DATASET=${TYPE}/fdb1k_${imsize}x_egl
    elif [ "$arg1" == "d" ]; then
        export DATASET=${TYPE}/fdb1k_${imsize}x_egl
    elif [ "$arg1" == "f" ]; then
        export DATASET=${TYPE}/fdb1k_${imsize}x_90jpeg_qualiy.ffcv
        # /local/43134625.1.gpu/ffcv/fdb1k_32x_90jpeg_qualiy.ffcv -f
    elif [ "$arg1" == "w" ]; then
        export DATASET="${TYPE}/fdb1k_${imsize}x_egl/fdb1k_${imsize}x_egl-train-{000000..000099}.tar"
    else
        echo "Unknown command: $arg1"
    fi
    

    # Train
    mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
    python pretrain_speed.py --data-dir ${DATASET} ${PIPE_FLAGS} \
    --model deit_${MODEL}_patch16_224 --experiment ${JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${CLS}k_${RENDER_HWD}_lr${LR}_ep${EPOCHS}_bs${BATCH_SIZE}_${STORAGE}_${EXPERIMENT} \
    --input-size 3 ${INPUT_SIZE} ${INPUT_SIZE} \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5  --color-jitter 0.4 \
    --hflip 0.5 --vflip 0.5 --scale 0.08 1.0 --ratio 0.75 1.3333 \
    --epochs ${EPOCHS} --opt adamw --lr ${LR} --weight-decay 0.05 --deit-scale 512.0 \
    --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 \
    --aa rand-m9-mstd0.5-inc1  --train-interpolation random \
    --reprob 0.25 --remode pixel \
    --batch-size ${LOCAL_BATCH_SIZE} -j 19 --pin-mem \
    --mixup 0.8 --cutmix 1.0 --drop-path 0.1 \
    --num-classes ${CLS}000 --eval-metric loss \
    --interval-saved-epochs 100 --output ${OUT_DIR} \
    --no-prefetcher --amp \

    end_time=$(date +%s%3N)
    total_duration=$((end_time - start_time))
    times+=("$total_duration")

    
    echo "______Finish_________"
    echo "                   "
done

echo "                   "
echo " Arguments : $arg1  $arg2      "

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