#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/$JOB_ID_ffcv_create_db.out
#$ -N build_ffcv

cat $JOB_SCRIPT
echo "JOB ID: ---- >>>>>>   $JOB_ID"
# ======== Modules ========
source /etc/profile.d/modules.sh
module purge
module load cuda/12.0/12.0.0 cudnn/8.8/8.8.1 nccl/2.17/2.17.1-1 gcc/12.2.0 cmake/3.26.1 singularitypro/3.9
# From NVIDIA
module load hpcx-mt/2.12

# ======== Pyenv/ ========
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
pyenv local anaconda3-2023.07-2/envs/ffcv


echo "Copy and Untar..."

time cp /home/acc12930pb/working/graphics/pyGL/volta_render_test_ist/data_362/FractalDB-1000-EGL-GLFW.tar $SGE_LOCALDIR
time tar -xf $SGE_LOCALDIR/FractalDB-1000-EGL-GLFW.tar -C $SGE_LOCALDIR
# time pv /beeond/FractalDB-1000-EGL-GLFW.tar | tar -x -C /beeond
echo "Finished copying and Untar..."

export DATA_DIR_IN=$SGE_LOCALDIR/FractalDB-1000-EGL-GLFW

export MAX_RES=362
# Compress probaility = 0.25 -> # Compress a random 1/4 of the dataset.
export COMP_PROB=0.10 
export QUALITY=1

python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}

export QUALITY=10
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=20
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=30
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=40
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=50
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=60
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=70
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=80
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=90
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}
export QUALITY=99
python write_datasets.py --cfg.dataset=fdb  --cfg.split=train \
     --cfg.data_dir=${DATA_DIR_IN} --cfg.write_path=./compress/fdb1k_${QUALITY}.ffcv \
     --cfg.max_resolution=${MAX_RES} --cfg.write_mode=proportion \
     --cfg.compress_probability=${COMP_PROB} --cfg.jpeg_quality=${QUALITY}

     
echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#