#!/bin/bash
#$ -cwd
#$ -l rt_F=100
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/$JOB_ID_pretrain_deit_base_fdb50k_wds.out
#$ -N pret_vit_base_patch16_224_pyto_wds
#####$ -l USE_BEEOND=1
cat $JOB_SCRIPT
echo "....................................................................................."
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

############# Render to local SSD
export LOCALDIR=/beeond
export RENDER_HWD=files
export DATASET=/NOT/WORKING

# cd render_engines/fdb

############ For WDS
export SHARDS="/home/acc12930pb/scratch/fdb/fdb50k/shards_fdb50k_egl/fdb50k-train-{000000..004999}.tar"

##################################

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=2042
export NGPUS=400
export NUM_PROC=4
export PIPE=PyTo
export STORAGE=wds

export MODEL=tiny
export LR=1.0e-3
export CLS=50
export EPOCHS=5
export LOCAL_BATCH_SIZE=32
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224

export EXPERIMENT=wds_50k

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${CLS}k/pre_training

wandb enabled
# FDB - 1k - Custom
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python pretrain.py /NOT/WORKING -w --trainshards ${SHARDS} \
    --model deit_${MODEL}_patch16_${INPUT_SIZE} --experiment ${JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${CLS}k_${RENDER_HWD}_lr${LR}_ep${EPOCHS}_bs${BATCH_SIZE}_${STORAGE}_${EXPERIMENT} \
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
    --no-prefetcher --amp  \
    --log-wandb \

echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

