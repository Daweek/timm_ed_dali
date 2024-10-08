#!/bin/sh
#$ -l rt_F=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/$JOB_ID_Tar1k
#$ -cwd
#$ -N tar_wds_1k

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

pyenv local torch_240_3124
#echo $SGE_TASK_ID

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 80 -np 80 \
       python makeshard_with_tarwriter.py \
        --splits train \
        --maxcount 10000 \
        --inputdata-path /home/acc12930pb/scratch/fdb/fdb1k/fdb1k_egl \
        --outshards-path /home/acc12930pb/scratch/fdb/fdb1k/shards_fdb1k_egl \
        --base-name fdb1k


echo "                   "
echo "                   "
echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

