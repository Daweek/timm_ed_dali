#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=10:00:00
#$ -j y
#$ -o output/$JOB_ID_finetune_pyto_deit_tiny_cifar10.out
#$ -N finetune_pyto_deit_tiny_cifar10

cat $JOB_SCRIPT
echo "..................................................................................................."
echo "JOB ID: ---- >>>>>>   $JOB_ID"
# ======== Modules ========
source /etc/profile.d/modules.sh
module purge
module load cuda/12.2/12.2.0 cudnn/8.9/8.9.2 nccl/2.18/2.18.5-1 gcc/12.2.0 cmake/3.26.1 hpcx-mt/2.12

# ======== Pyenv/ ========
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
pyenv local torch_21_3117

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# ========== For MPI
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=2042
export NGPUS=8
export NUM_PROC=4
export PIPE=PyTo

# ========= For experiment and pre-train
export RENDER_HWD=files
export PRE_STORAGE=ssd
export MODEL=tiny
export PRE_CLS=1
export PRE_LR=1.0e-3
export PRE_EPOCHS=300
export PRE_BATCH=512
export BATCH_SIZE=768
export LOCAL_BATCH_SIZE=96

# ========= Fine-Tune dataset info
export DATASET_NAME=cifar10
export DATASET_NUMCLS=10

export SSD=/local/${JOB_ID}.1.gpu
export PRE_JOB_ID=00000000
export PRE_EXPERIMENT=GLFW3090

export EXPERIMENT=GLFW3090
# For Timm scripts...
export CP_DIR=/home/acc12930pb/working/transformer/beforedali_timm_main_sora/checkpoint/tiny/fdb1k/pre_training/pretrain_deit_tiny_fdb1k_lr1.0e-3_epochs300_bs512_ssd_362x_GLFW3090/last.pth.tar  #----->>>>> best so far... 86.72

# export CP_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/pre_training/${PRE_JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_ep${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${PRE_EXPERIMENT}/last.pth.tar

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/fine_tuning

echo "Copy and Untar..."
mpirun --display-map --display-allocation --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 2 time tar -xf /home/acc12930pb/datasets/cifar10.tar -C ${SSD}
readlink -f ${SSD}
echo "Finished copying and Untar..."

wandb enabled

mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python finetune.py ${SSD}/cifar10 \
    --model deit_${MODEL}_patch16_224 --experiment ${JOB_ID}_fine_deit_${PIPE}_${MODEL}_${DATASET_NAME}_from_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${EXPERIMENT} \
    --input-size 3 224 224 --num-classes ${DATASET_NUMCLS}  \
    --batch-size ${LOCAL_BATCH_SIZE} --opt sgd --lr 0.01 --weight-decay 0.0001 --deit-scale 512.0 \
    --sched cosine  --epochs 1000  --lr-cycle-mul 1.0 --min-lr 1e-05 --decay-rate 0.1 --warmup-lr 1e-06 --warmup-epochs 10  --lr-cycle-limit 1 --cooldown-epochs 0 \
    --scale 0.08 1.0 --ratio 0.75 1.3333 \
    --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --mixup-mode batch --smoothing 0.1 --drop-path 0.1 \
    -j 19 --no-prefetcher \
    --output ${OUT_DIR} \
    --log-wandb \
    --pin-mem \
    --pretrained-path ${CP_DIR} \
    --amp \

echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

