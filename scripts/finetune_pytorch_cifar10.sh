#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:ncpus=96:ngpus=8:mpiprocs=8
#PBS -N ft_cifar100
#PBS -l walltime=02:00:00
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

# source ./config.sh
# ======== Modules ========
echo "Include main ABCI modules for 3.0 .."
source /etc/profile.d/modules.sh
module purge
####### MPI
# module load intel-mpi/2021.13
module load hpcx/2.20
# module load hpcx-mt/2.20
# Load CUSTOM OpenMPI with CUDA support
# export PATH=$HOME/apps/openmpi/bin:$PATH
module load cuda/12.4/12.4.1 cudnn/9.5/9.5.1 nccl/2.23/2.23.4-1 nvhpc/24.9 gdrcopy/2.4.1 

# ======== Pyenv/ ========
echo "Include main python environment..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

pyenv local 3.11.11

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"



######################################################################################

# ========== For MPI
# export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_ADDR=$(
  ip a show dev bond0 \
  | grep 'inet ' \
  | head -n 1 \
  | cut -d " " -f 6 \
  | cut -d "/" -f 1
)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"
# export NGPUS=8
# export NUM_PROC=16
export PIPE=PyTo

# ========= For experiment and pre-train
export RENDER_HWD=cpu
export PRE_STORAGE=ssd
export MODEL=tiny
export PRE_CLS=21
export PRE_LR=1.0e-3
export PRE_EPOCHS=90
export PRE_BATCH=8960

export BATCH_SIZE=768
export LOCAL_BATCH_SIZE=96

# ========= Fine-Tune dataset info
export DATASET_NAME=cifar10
export DATASET_NUMCLS=10

export SSD=$PBS_LOCALDIR
    echo "LOCAL_SSD: ${SSD}"
export PRE_JOB_ID=42084625
export PRE_EXPERIMENT=localShuf

export EXPERIMENT=localShuf
# For Timm scripts...
# export CP_DIR=/home/acc12930pb/working/transformer/beforedali_timm_main_sora/checkpoint/tiny/fdb1k/pre_training/pretrain_deit_tiny_fdb1k_lr1.0e-3_epochs300_bs512_ssd_362x_GLFW3090/last.pth.tar  #----->>>>> best so far... 86.72

# export CP_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/pre_training/${PRE_JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_ep${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${PRE_EXPERIMENT}/last.pth.tar

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/fine_tuning


#### From others
NUM_GPU_PER_NODE=8
NUM_NODES=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"



# echo "Copy and Untar..."
# mpirun --bind-to socket -machinefile $PBS_NODEFILE -npernode 1 -np 2 time tar -xf /groups/gcc50533/edgar/tared_datasets/cifar10.tar -C ${SSD}/
# readlink -f ${SSD}/cifar10
# ls ${SSD}/ |wc -l
# echo "Finished copying and Untar..."

# wandb enabled

# Checking the hostfile
mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

mpirun -np ${NUM_GPUS} \
    -hostfile $HOSTFILE_NAME\
    --npernode ${NUM_GPU_PER_NODE} \
    -x MASTER_ADDR=${MASTER_ADDR} \
    -x MASTER_PORT=${MASTER_PORT} \
    -bind-to none \
    -x PATH \
    -x MPI_HOME \
    -x OMPI_HOME \
    python finetune.py /home/acc12930pb/datasets/cifar10 \
        --model deit_${MODEL}_patch16_224 --experiment ${JOB_ID}_fine_deit_${PIPE}_${MODEL}_${DATASET_NAME}_from_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${EXPERIMENT} \
        --input-size 3 224 224 --num-classes ${DATASET_NUMCLS}  \
        --batch-size ${LOCAL_BATCH_SIZE} --opt sgd --lr 0.01 --weight-decay 0.0001 --deit-scale 512.0 \
        --sched cosine  --epochs 1000  --lr-cycle-mul 1.0 --min-lr 1e-05 --decay-rate 0.1 --warmup-lr 1e-06 --warmup-epochs 10  --lr-cycle-limit 1 --cooldown-epochs 0 \
        --scale 0.08 1.0 --ratio 0.75 1.3333 \
        --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --mixup-mode batch --smoothing 0.1 --drop-path 0.1 \
        -j 4 --no-prefetcher \
        --output ${OUT_DIR} \
        --pin-mem \
        --amp \

    # --pretrained-path ${CP_DIR} \

echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

