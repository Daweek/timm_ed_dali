#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=192
#PBS -N LP_cifar10
#PBS -l walltime=02:00:00
#PBS -P gah51624
#PBS -j oe
#PBS -V
#PBS -koed
#PBS -o output/

# =========== Configuration =========================================================
source $HOME/utils/main_config.sh
pyenv local 3.12.2

# ========= Get local Directory ======================================================
if [  "$PBS_O_WORKDIR" = "$HOME" ]; then
    cecho gray "PBS_O_WORKDIR is set to HOME. Using current directory."
    PBS_O_WORKDIR=$(pwd)
else
    cecho gray "PBS_O_WORKDIR is set to: $PBS_O_WORKDIR"
fi
# ======== Modules and Python on main .configure.sh ==================================
cecho gray "Changing to working directory: $PBS_O_WORKDIR"
cd $PBS_O_WORKDIR

# =========== Basic Information =========================================================
blank_lines 2
cecho green "Job started on: $(date)"
cecho green "ABCI 3.0 ............................................................"
JOB_ID=$(echo "${PBS_JOBID:-$$}"  | cut -d '.' -f 1)
cecho green "JOB ID: ------- >>>>>>  $JOB_ID"
cecho red   "Hostname:------ >>>>>>  $(hostname)"

# =========== Create Output Directory and Experiments ARGS ============================================
blank_lines 2
# ========== For MPI
#### From others
NUM_GPU_PER_NODE=8
NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"
echo "NUM_NODES: ${NUM_NODES}"


export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"

# ========= For experiment and pre-train
export PIPE=PyTo
export RENDER_HWD=files
export PRE_STORAGE=ssd
export MODEL=tiny
export PRE_CLS=1
export PRE_LR=1.0e-3
export PRE_EPOCHS=80k
export PRE_BATCH=256

export BATCH_SIZE=768
export LOCAL_BATCH_SIZE=96

# ========= Fine-Tune dataset info
export DATASET_NAME=cifar10
export DATASET_NUMCLS=10

export SSD=$PBS_LOCALDIR
    echo "LOCAL_SSD: ${SSD}"
export PRE_JOB_ID=42084625
export PRE_EXPERIMENT=localShuf

export EXPERIMENT=LP_CIFAR10

# =========== Start of Job =========================================================
blank_lines 4
cecho orange "##### START ##############"
cecho orange "______Start Computing_________"
cecho orange "Job Started on: $(date)"
start_time=$(date +%s%3N)


###################################### Tar to SSD
echo "Copy and Untar..."
mpirun --mca btl tcp,smcuda,self -np 1 -map-by ppr:${NUM_NODES}:node -hostfile $PBS_NODEFILE tar -xf /home/acc12930pb/groups_shared/datasets/cifar10.tar -C $PBS_LOCALDIR
readlink -f ${SSD}/cifar10
ls ${SSD}/ 
echo "Finished copying and Untar..."

wandb enabled


# For Timm scripts...
# export CP_DIR=/home/acc12930pb/working/transformer/beforedali_timm_main_sora/checkpoint/tiny/fdb1k/pre_training/pretrain_deit_tiny_fdb1k_lr1.0e-3_epochs300_bs512_ssd_362x_GLFW3090/last.pth.tar  #----->>>>> best so far... 86.72

# export CP_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/pre_training/${PRE_JOB_ID}_pret_deit_${PIPE}_${MODEL}_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_ep${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${PRE_EXPERIMENT}/last.pth.tar

export CP_DIR=/home/acc12930pb/working/transformer/nakamura/1p-frac/train/output/pretrain/vit_tiny_patch16_224_sigma3.5_delta0.1_sample1000_80000ep.pth

export OUT_DIR=/home/acc12930pb/working/transformer/timm_ed_dali/checkpoint/${MODEL}/fdb${PRE_CLS}k/fine_tuning

# mpirun --use-hwthread-cpus --oversubscribe -np ${NUM_GPUS}  \
mpirun -np ${NUM_GPUS} -hostfile $PBS_NODEFILE --bind-to socket --oversubscribe -map-by ppr:8:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 -x MASTER_ADDR=${MASTER_ADDR} -x MASTER_PORT=${MASTER_PORT} \
        python finetune.py ${SSD}/cifar10 \
        --model deit_${MODEL}_patch16_224 --experiment ${JOB_ID}_fine_deit_${PIPE}_${MODEL}_${DATASET_NAME}_from_fdb${PRE_CLS}k_${RENDER_HWD}_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_${PRE_STORAGE}_${EXPERIMENT} \
        --input-size 3 224 224 --num-classes ${DATASET_NUMCLS}  \
        --batch-size ${LOCAL_BATCH_SIZE} --opt sgd --lr 0.01 --weight-decay 0.0001 --deit-scale 512.0 \
        --sched cosine  --epochs 1000  --lr-cycle-mul 1.0 --min-lr 1e-05 --decay-rate 0.1 --warmup-lr 1e-06 --warmup-epochs 10  --lr-cycle-limit 1 --cooldown-epochs 0 \
        --scale 0.08 1.0 --ratio 0.75 1.3333 \
        --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --mixup-mode batch --smoothing 0.1 --drop-path 0.1 \
        -j 23 --no-prefetcher \
        --output ${OUT_DIR} \
        --pin-mem \
        --amp \
        --pretrained-path ${CP_DIR} \

# =========== End of RUNNING ============================================================

blank_lines 4

end_time=$(date +%s%3N)
total_duration=$((end_time - start_time))

cecho orange "This experiment Duration: "
convert_milliseconds "$total_duration"
cecho orange "JOB ID: ------- >>>>>>  $JOB_ID"
cecho red    "Hostname:------ >>>>>>  $(hostname)"
cecho bold magenta "Experiment: $EXPERIMENT"
cecho orange "Job finished on: $(date)"
cecho orange "______Finish_________"
echo "                   "

# =========== End of Job ========================================================
## Debuggin purposes???
cecho cyan "code=$?"

