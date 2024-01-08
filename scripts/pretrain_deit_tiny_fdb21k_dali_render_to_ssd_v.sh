#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=30:00:00
#$ -j y
#$ -o output/$JOB_ID_pretrain_deit_tiny_dali_fdb21k_rendertossd.out
#$ -N pretrain_vit_tiny_dali_patch16_224
#$ -l USE_BEEOND=1
cat $JOB_SCRIPT
cat dali/pipe_train.py
echo "....................................................................................."
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

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

############# Render to local SSD
export LOCALDIR=/beeond
export RENDER_HWD=egl
export DATASET=${LOCALDIR}/fdb21k_${RENDER_HWD}

cd render_engines/fdb

echo "Start SEARCHING to local ..."
# mpirun --bind-to none --use-hwthread-cpus -np 80 python mpi_cpu.py --save_root ${LOCALDIR}/fdb1k_cpu
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 80 -np 2560 python mpi_ifs_search_egl.py --category 21000 --save_dir /beeond

echo "Start rendering to local ..."
# mpirun --bind-to none --use-hwthread-cpus -np 80 python mpi_cpu.py --save_root ${LOCALDIR}/fdb1k_cpu
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 80 -np 2560 python mpi_gpu.py --ngpus-pernode 4 --image_res 362 --save_root /beeond/fdb21k --load_root /beeond/csv_rate0.2_category21000_points200000 
# du -sh ${DATASET}
cd ../../
##################################

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=2042
export NGPUS=128
export NUM_PROC=4
export PIPE=Dali
export STORAGE=ssd

export MODEL=tiny
export LR=8.0e-3
export CLS=21
export EPOCHS=90
export LOCAL_BATCH_SIZE=64
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224

export EXPERIMENT=searchCSV

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${CLS}k/pre_training


wandb enabled

# FDB - 1k - Custom
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python pretrain.py ${DATASET} --dali \
    --model deit_${MODEL}_patch16_${INPUT_SIZE} --experiment ${JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${CLS}k_${RENDER_HWD}_lr${LR}_ep${EPOCHS}_bs${BATCH_SIZE}_${STORAGE}_${EXPERIMENT} \
    --input-size 3 ${INPUT_SIZE} ${INPUT_SIZE} \
    --epochs ${EPOCHS} --opt adamw --lr ${LR} --weight-decay 0.05 --deit-scale 512.0 \
    --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 \
    --batch-size ${LOCAL_BATCH_SIZE} \
    --scale 0.08 1.0 --ratio 0.75 1.3333 \
    --num-classes ${CLS}000 --eval-metric loss \
    --mixup 0.8 --cutmix 1.0 --drop-path 0.1 \
    -j 19 --pin-mem \
    --interval-saved-epochs 100 --output ${OUT_DIR} \
    --no-prefetcher \
    --log-wandb \
    --amp \

### For RCDB21k SORA BS-> 32,768 for model : Base
# pretrain.py /NOT/WORKING -w --trainshards /bb/grandchallenge/gae50969/datasets/ExRCDB-21k_v4_Shards/rcdb_21k_v4-train-{000000..002099}.tar --model deit_base_patch16_224 --experiment pretrain_deit_base_RCDB21k_nami_v4_lr4.0e-3_epochs90_bs32768_CVPR2022_amp_shard_clipping_512GPUs --input-size 3 224 224 --aa rand-m9-mstd0.5-inc1 --hflip 0.0 --interpolation bicubic --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --reprob 0.25 --remode pixel --batch-size 64 -j 4 --pin-mem --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --opt adamw --lr 4.0e-3 --weight-decay 0.05 --epochs 90 --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1.0e-6 --warmup-iter 5000 --cooldown-epochs 0 --num-classes 21000 --eval-metric loss --no-prefetcher --interval-saved-epochs 10 --output /bb/grandchallenge/gae50969/yokota_check_points/base/rcdb21k/pre_training --clip-grad 0.25 --resume /bb/grandchallenge/gae50969/yokota_check_points/base/rcdb21k/pre_training/pretrain_deit_base_RCDB21k_nami_v4_lr4.0e-3_epochs90_bs32768_CVPR2022_amp_shard_clipping_512GPUs/checkpoint-45.pth.tar --log-wandb
### For RCDB21k SORA BS-> 8,192
# pretrain.py /NOT/WORKING -w --trainshards /bb/grandchallenge/gae50969/datasets/ExRCDB-21k_Shards/rcdb_21k-train-{000000..002099}.tar --model deit_base_patch16_224 --experiment pretrain_deit_base_RCDB21k_nami_lr1.0e-3_epochs90_bs8192_CVPR2022_amp_shard_lwn_clipping_512GPUs --input-size 3 224 224 --aa rand-m9-mstd0.5-inc1 --hflip 0.0 --interpolation bicubic --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --reprob 0.25 --remode pixel --batch-size 16 -j 1 --pin-mem --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --opt adamw --lr 1.0e-3 --weight-decay 0.05 --epochs 90 --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1.0e-6 --warmup-iter 5000 --cooldown-epochs 0 --num-classes 21000 --eval-metric loss --no-prefetcher --interval-saved-epochs 10 --output /bb/grandchallenge/gae50969/yokota_check_points/base/rcdb21k/pre_training --layer-decay 0.75 --resume /bb/grandchallenge/gae50969/yokota_check_points/base/rcdb21k/pre_training/pretrain_deit_base_RCDB21k_nami_lr1.0e-3_epochs90_bs8192_CVPR2022_amp_shard_lwn_512GPUs/last.pth.tar --log-wandb

echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

