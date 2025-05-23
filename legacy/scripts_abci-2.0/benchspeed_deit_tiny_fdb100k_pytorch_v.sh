#!/bin/bash
#$ -cwd
#$ -l rt_F=64
#$ -l h_rt=20:00:00
#$ -j y
#$ -o output/$JOB_ID_benchspeed_deit_tiny_pyto_fdb100k.out
#$ -N benchspeed_vit_tiny_pyto_patch16_224
#####$ -l USE_BEEOND=1
cat $JOB_SCRIPT
echo "................................................................................"
echo "JOB ID: ---- >>>>>>   $JOB_ID"

# ======== Modules and Python on main .configure.sh ==================================

source ./config.sh

######################################################################################

############# Render to local SSD
# export LOCALDIR=/beeond
export RENDER_HWD=files
export DATASET=/home/acc12930pb/scratch/fdb/fdb100k/fdb100k_egl

# cd render_engines/fdb

# echo "Start SEARCHING to local ..."
# # mpirun --bind-to none --use-hwthread-cpus -np 80 python mpi_cpu.py --save_root ${LOCALDIR}/fdb1k_cpu
# mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 80 -np 2560 python mpi_ifs_search_egl.py --ngpus-pernode 4 --category 21000 --save_dir /beeond

# echo "Start rendering to local ..."
# # mpirun --bind-to none --use-hwthread-cpus -np 80 python mpi_cpu.py --save_root ${LOCALDIR}/fdb1k_cpu
# mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 80 -np 2560 python mpi_gpu.py --ngpus-pernode 4 --image_res 362 --save_root /beeond/fdb21k --load_root /beeond/csv_rate0.2_category21000_points200000 
# # du -sh ${DATASET}
# cd ../../
##################################

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=2042
export NGPUS=256
export NUM_PROC=4
export PIPE=PyTo
export STORAGE=nfs

export MODEL=tiny
export LR=1.0e-3
export CLS=100
export EPOCHS=4
export LOCAL_BATCH_SIZE=32
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224

export EXPERIMENT=speed_nfs

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${CLS}k/pre_training


wandb enabled

# FDB - 1k - Custom
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python pretrain.py ${DATASET} \
    --model deit_${MODEL}_patch16_${INPUT_SIZE} --experiment ${JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${CLS}k_${RENDER_HWD}_lr${LR}_ep${EPOCHS}_bs${BATCH_SIZE}_${STORAGE}_${EXPERIMENT} \
    --input-size 3 ${INPUT_SIZE} ${INPUT_SIZE} \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5  --color-jitter 0.4 \
    --hflip 0.5 --vflip 0.5 --scale 0.08 1.0 --ratio 0.75 1.3333 \
    --epochs ${EPOCHS} --opt adamw --lr ${LR} --weight-decay 0.05 --deit-scale 8192.0 \
    --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 \
    --aa rand-m9-mstd0.5-inc1  --train-interpolation random \
    --reprob 0.25 --remode pixel \
    --batch-size ${LOCAL_BATCH_SIZE} -j 19 --pin-mem \
    --mixup 0.8 --cutmix 1.0 --drop-path 0.1 \
    --num-classes ${CLS}000 --eval-metric loss \
    --interval-saved-epochs 100 --output ${OUT_DIR} \
    --no-prefetcher  \
    --log-wandb \
    --amp \


echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

