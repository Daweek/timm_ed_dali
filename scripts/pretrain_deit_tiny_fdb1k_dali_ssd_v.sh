#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=15:00:00
#$ -j y
#$ -o output/$JOB_ID_pretrain_deit_tiny_fdb1k_ssd.out
#$ -N pretrain_vit_tiny_patch16_224
#$ -l USE_BEEOND=1
cat $JOB_SCRIPT
echo ".....................................................................................\n\n\n"
echo "JOB ID: ---- >>>>>>   $JOB_ID"
# ======== Modules ========
source /etc/profile.d/modules.sh
module purge
module load cuda/12.0/12.0.0 cudnn/8.8/8.8.1 nccl/2.17/2.17.1-1 gcc/12.2.0 cmake/3.26.1 hpcx-mt/2.12

# ======== Pyenv/ ========
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
pyenv local torch_20_311

wandb enabled

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=2042

export MODEL=tiny
export LR=1.0e-3
export CLS=1
export EPOCHS=300
export OUT_DIR=/home/acc12930pb/working/transformer/timm_main_sora/checkpoint/${MODEL}/fdb${CLS}k/pre_training

export NGPUS=16
export NUM_PROC=4
export LOCAL_BATCH_SIZE=32
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224


echo "Copy and Untar..."

time cp /home/acc12930pb/working/graphics/pyGL/volta_render_test_ist/data_362/FractalDB-1000-EGL-GLFW.tar /beeond
time tar -xf /beeond/FractalDB-1000-EGL-GLFW.tar -C /beeond
# time pv /beeond/FractalDB-1000-EGL-GLFW.tar | tar -x -C /beeond
echo "Finished copying and Untar..."

# FDB - 1k - Custom
mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python pretrain_dali.py /beeond/FractalDB-1000-EGL-GLFW --dali \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_Dali_${MODEL}_fdb${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_ssd_2interp_x324_V \
    --input-size 3 ${INPUT_SIZE} ${INPUT_SIZE} \
    --epochs ${EPOCHS} --opt adamw --lr ${LR} --weight-decay 0.05 --deit-scale 512.0 \
    --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 \
    --batch-size ${LOCAL_BATCH_SIZE} \
    --scale 0.08 1.0 --ratio 0.75 1.3333 \
    --num-classes ${CLS}000 --eval-metric loss \
    --mixup 0.8 --cutmix 1.0 --drop-path 0.1 \
    -j 16 --pin-mem \
    --interval-saved-epochs 100 --output ${OUT_DIR} \
    --no-prefetcher --amp \
    --log-wandb \

echo "                   "
echo "                   "
echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

