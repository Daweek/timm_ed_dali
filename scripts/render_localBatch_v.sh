#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=00:20:00
#$ -j y
#$ -o output/$JOB_ID_rendertossd_BatchLocal.out
#$ -N renderonly
##################$ -l USE_BEEOND=1
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

############# Render to local SSD
export SSD=/local/${JOB_ID}.1.gpu
export LOCALDIR=${SSD}
export RENDER_HWD=egl
export DATASET=${LOCALDIR}/fdb1k_${RENDER_HWD}

cd render_engines/fdb
echo "Start SEARCHING to local ..."

# Frist copy the local searched scv file to SSD
mpirun --display-map --display-allocation --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 4 time tar -xf /home/acc12930pb/working/transformer/timm_ed_dali/render_engines/fdb/csv/data1k_fromPython.tar -C ${SSD}

readlink -f ${SSD}
ls ${SSD}
# mpirun --bind-to none --use-hwthread-cpus -np 80 python mpi_cpu.py --save_root ${LOCALDIR}/fdb1k_cpu

### This is for searching using all the reources....
# mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 80 -np 160 python mpi_ifs_search_egl.py --ngpus-pernode 4 --category 1000 --save_dir /beeond

echo "Start REDNERING to local ..."

mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 4 mkdir ${SSD}/fdb1k_egl

mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 50 -np 200 python mpi_gpu.py --image_res 362 --ngpus-pernode 4 --save_root ${SSD}/fdb1k --load_root ${SSD}/data1k_fromPython/csv_rate0.2_category1000 

##### Debug local
readlink -f ${SSD}
ls ${SSD}

mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 4 ls ${SSD}/fdb1k_egl 

mpirun --bind-to none -machinefile $SGE_JOB_HOSTLIST -npernode 1 -np 4 find ${SSD}/fdb1k_egl -type f -print |wc -l



# ${SSD}/fdb1k
# du -sh ${DATASET}

cd ../../
##################################

# echo "Debug Finished..."
# exit 0

echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#

