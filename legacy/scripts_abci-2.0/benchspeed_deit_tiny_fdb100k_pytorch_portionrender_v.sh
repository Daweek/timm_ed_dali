#!/bin/bash
#$ -cwd
#$ -l rt_F=100
#$ -l h_rt=15:00:00
#$ -j y
#$ -o output/$JOB_ID_pretrain_deit_tiny_pyto_fdb100k_renderportiontossd.out
#$ -N pret_vit_tiny_p16_224_pyto_fdb100k_portion
#$ -l USE_BEEOND=1

cat $JOB_SCRIPT
echo "................................................................................"
echo "JOB ID: ---- >>>>>>   $JOB_ID"

# ======== Modules and Python on main .configure.sh ==================================

source ./config.sh

######################################################################################

############# Render to local SSD
export SSD=/local/${JOB_ID}.1.gpu
export LOCALDIR=${SSD}
export RENDER_HWD=cpu
export DATASET=${LOCALDIR}/fdb100k_${RENDER_HWD}

cd render_engines/fdb
echo "Start SEARCHING to local ..."
# mpirun --bind-to none --use-hwthread-cpus -np 80 python mpi_cpu.py --save_root ${LOCALDIR}/fdb1k_cpu
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 50 -np 5000 python mpi_ifs_search_egl.py --ngpus-pernode 4 --category 100000 --save_dir /beeond
# mpirun --display-map --display-allocation --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 4 time tar -xf /home/acc12930pb/working/transformer/timm_ed_dali/render_engines/fdb/csv/data1k_fromPython.tar -C ${SSD}

# ls /beeond
# ls /beeond/csv_rate0.2_category10000_points200000
# readlink -f ${SSD}
# ls ${SSD}
# ls /local

echo "Make sure that we create a directory before we start rendering on each node..."
# For GPU
# mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 4 mkdir ${SSD}/fdb1k_egl
# For CPU
mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 100 mkdir ${SSD}/fdb100k_cpu

echo "Start RENDERING to local ..."
# For GPU
# mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 50 -np 200 python mpi_gpu.py --image_res 362 --ngpus-pernode 4 --save_root ${SSD}/fdb1k --load_root ${SSD}/data1k_fromPython/csv_rate0.2_category1000 
# For CPU
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 50 -np 5000 python mpi_cpu.py --image_res 362 --save_root ${SSD}/fdb100k_cpu --load_root /beeond/csv_rate0.2_category100000_points200000


##### Debug local
readlink -f ${SSD}
ls ${SSD}
mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 100 ls ${SSD}/fdb100k_cpu 
mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 100 find ${SSD}/fdb100k_cpu -type f -print |wc -l

cd ../../
##################################

# echo "Debug Finished..."
# exit 0

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=2042
export NGPUS=400
export NUM_PROC=4
export PIPE=PyTo
export STORAGE=ssd

export MODEL=tiny
export LR=1.0e-3
export CLS=100
export EPOCHS=90
export LOCAL_BATCH_SIZE=32
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224

export EXPERIMENT=speed_rt_100n

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${CLS}k/pre_training

wandb enabled

# FDB - 1k - Custom
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python portiontossd_pretrain.py ${DATASET} \
    --model deit_${MODEL}_patch16_224 --experiment ${JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${CLS}k_${RENDER_HWD}_lr${LR}_ep${EPOCHS}_bs${BATCH_SIZE}_${STORAGE}_${EXPERIMENT} \
    --input-size 3 ${INPUT_SIZE} ${INPUT_SIZE} \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5  --color-jitter 0.4 \
    --hflip 0.5 --vflip 0.5 --scale 0.08 1.0 --ratio 0.75 1.3333 \
    --epochs ${EPOCHS} --opt adamw --lr ${LR} --weight-decay 0.05 --deit-scale 12800.0 \
    --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 \
    --aa rand-m9-mstd0.5-inc1  --train-interpolation random \
    --reprob 0.25 --remode pixel \
    --batch-size ${LOCAL_BATCH_SIZE} -j 19 --pin-mem \
    --mixup 0.8 --cutmix 1.0 --drop-path 0.1 \
    --num-classes ${CLS}000 --eval-metric loss \
    --interval-saved-epochs 100 --output ${OUT_DIR} \
    --no-prefetcher --amp \
    --log-wandb \
    --portiontossd --portion-ngpus 4 \

echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

