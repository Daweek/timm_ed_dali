#!/bin/bash
#PBS -q rt_HF
#PBS -l select=2:ncpus=384:ngpus=16:mpiprocs=192
#PBS -N ft_imnet1k
#PBS -l walltime=06:30:00
#PBS -P gcc50533
#PBS -j oe
#PBS -V
#PBS -koed
#PBS -o output/

# cat $JOB_SCRIPT
echo "ABCI 3.0 ..................................................................................."
JOB_ID=$(echo "${PBS_JOBID}" | cut -d '.' -f 1)
echo "JOB ID: ---- >>>>>>  $JOB_ID"

# ========= Get local Directory ======================================================
cd $PBS_O_WORKDIR
pwd -LP
# ======== Modules and Python on main .configure.sh ==================================
source ./config.sh
######################################################################################
# ========== For MPI
#### From others
NUM_GPU_PER_NODE=8
NUM_NODES=2
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"
echo "NUM_NODES: ${NUM_NODES}"


export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"

# ========= For experiment and pre-train
export PIPE=Dali
export RENDER_HWD=files
export PRE_STORAGE=ssd
export MODEL=tiny
export PRE_CLS=0
export PRE_LR=1.0e-3
export PRE_EPOCHS=0
export PRE_BATCH=0

export LOCAL_BATCH_SIZE=32
export BATCH_SIZE=$(($NUM_GPUS * $LOCAL_BATCH_SIZE))

# ========= Fine-Tune dataset info
export DATASET_NAME=imnet
export DATASET_NUMCLS=1000

export SSD=$PBS_LOCALDIR
    echo "LOCAL_SSD: ${SSD}"
export PRE_JOB_ID=0
export PRE_EXPERIMENT=none

export EXPERIMENT=Dali_h200_1

# For Timm scripts...
#----->>>>> best so far... 86.72
export CP_DIR=/home/acc12930pb/working/transformer/beforedali_timm_main_sora/checkpoint/tiny/fdb1k/pre_training/pretrain_deit_tiny_fdb1k_lr1.0e-3_epochs300_bs512_ssd_362x_GLFW3090/last.pth.tar  

# export CP_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/pre_training/${PRE_JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_ep${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${PRE_EXPERIMENT}/last.pth.tar

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/fine_tuning

###################################### Tar to SSD
echo "Copy and Untar..."
mpirun --mca btl tcp,smcuda,self -np ${NUM_NODES} -map-by ppr:1:node -hostfile $PBS_NODEFILE tar -xf /home/acc12930pb/groups_shared/datasets/imnet.tar -C $PBS_LOCALDIR
readlink -f ${SSD}/${DATASET_NAME}
ls ${SSD}/${DATASET_NAME}/train | wc -l
echo "Finished copying and Untar..."

# wandb enabled

mpirun -np ${NUM_GPUS} -hostfile $PBS_NODEFILE --bind-to socket --oversubscribe -map-by ppr:8:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 -x MASTER_ADDR=${MASTER_ADDR} -x MASTER_PORT=${MASTER_PORT} \
        python finetune.py $SSD/imnet --dali \
        --model deit_${MODEL}_patch16_224 --experiment ${JOB_ID}_fine_deit_${PIPE}_${MODEL}_${DATASET_NAME}_from_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${EXPERIMENT} \
        --input-size 3 224 224 --num-classes ${DATASET_NUMCLS}  \
        --batch-size ${LOCAL_BATCH_SIZE} --opt adamw --lr 0.001 --weight-decay 0.05 --deit-scale 512.0 \
        --sched cosine  --epochs 300  --lr-cycle-mul 1.0 --min-lr 1e-05 --decay-rate 0.1 --warmup-lr 1e-06 --warmup-epochs 10  --lr-cycle-limit 1 --cooldown-epochs 0 \
        --scale 0.08 1.0 --ratio 0.75 1.3333 --hflip 0.5 --color-jitter 0.4 --interpolation bicubic --train-interpolation bicubic --crop-pct 1.0 \
        --mean 0.485 0.456 0.406 \
        --std 0.229 0.224 0.225 \
        --reprob 0.5 --remode pixel \
        --aa rand-m9-mstd0.5-inc1 \
        --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --mixup-mode batch --smoothing 0.1 --drop-path 0.1 \
        -j 23 --no-prefetcher \
        --amp \
        --pin-mem \
        --output ${OUT_DIR} \
        --pretrained-path ${CP_DIR} \
        --log-wandb \

    # 

echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

