#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=15:00:00
#$ -j y
#$ -o output/$JOB_ID_pretrain_deit_tiny_pyto_fdb1k_ssd.out
#$ -N pret_vit_tiny_224_pyto_fdb1k
######$ -l USE_BEEOND=1
cat $JOB_SCRIPT
echo ".....................................................................................\n\n\n"
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
pyenv local torch_22_3121

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

############# Render to local SSD
export SSD=/local/${JOB_ID}.1.gpu
export LOCALDIR=${SSD}
export RENDER_HWD=files
export DATASET=${LOCALDIR}/FractalDB1k_1k_CPUNak

# echo "Copy and Untar..."

# # time cp /home/acc12930pb/working/graphics/pyGL/volta_render_test_ist/data_362/FractalDB1k_1k_CPUNak.tar /beeond
# # time tar -xf ${SSD}/FractalDB1k_1k_CPUNak.tar -C /beeond
# mpirun --display-map --display-allocation --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 4 time tar -xf /home/acc12930pb/working/graphics/pyGL/volta_render_test_ist/data_362/FractalDB1k_1k_CPUNak.tar -C ${SSD}
# readlink -f ${SSD}

# time pv /beeond/FractalDB-1000-EGL-GLFW.tar | tar -x -C /beeond
# echo "Finished copying and Untar..."

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=2042
export NGPUS=16
export NUM_PROC=4
export PIPE=PyTo
export STORAGE=ssd

export MODEL=tiny
export LR=1.0e-3
export CLS=1
export EPOCHS=10
export LOCAL_BATCH_SIZE=32
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224

export EXPERIMENT=CPURAM

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${CLS}k/pre_training

wandb enabled

# FDB - 1k - Custom
mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python pretrain.py /NOT/WORKING --render-type cpu --csv /home/acc12930pb/working/transformer/timm_ed_dali/render_engines/fdb/csv/searched_params/csv_rate0.2_category1000_points200000 --ram-batch \
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
    # --log-wandb \

echo "                   "
echo "                   "
echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

