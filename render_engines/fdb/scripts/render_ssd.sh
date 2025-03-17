#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=192
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
source ./../config.sh
######################################################################################
# ========== For MPI
#### From others
NUM_GPU_PER_NODE=8
NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"

# ========= For experiment and pre-train
#######################################################
global_start=$(date +%s%3N)
# Experiments parameters
size=(32 64 128 256 512 1024)
# size=(32)

# NFS network_________________
#root=nfs/raw

# SSD ___________________________
export SSD=${PBS_LOCALDIR}
root=${SSD}/raw
# Check if the directory exists
if [ -d "$root" ]; then
    echo "Directory '$root' already exists."
else
    echo "Directory '$root' does not exist. Creating it now..."
    mkdir "$root"
fi
#SSD___________________________________


for imsize in "${size[@]}"
do
    echo "______Start rendering_________"

    start_time=$(date +%s%3N)

    # Render to somewhere
    # mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -np 80 python mpi_gpu.py --ngpus-pernode 4 --image_res $imsize --save_root ${root}/fdb1k_${imsize}x 
    mpirun -np ${NUM_GPUS} -hostfile $PBS_NODEFILE --bind-to socket --oversubscribe -map-by ppr:8:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 -x MASTER_ADDR=${MASTER_ADDR} -x MASTER_PORT=${MASTER_PORT} python mpi_gpu.py --ngpus-pernode 8 --image_res $imsize --save_root ${root}/fdb1k_${imsize}x
    
    #--instance 10 --rotation 1 --nweights 1

    end_time=$(date +%s%3N)
    total_duration=$((end_time - start_time))
    times+=("$total_duration")

    du -sh ${root}/fdb1k_${imsize}x_egl

    echo "Number of images generated:"
    find ${root}/fdb1k_${imsize}x_egl -type f -print |wc -l | xargs printf "%'d\n"
    
    echo "This experiment Duration: ${imsize}x${imsize} "
    convert_milliseconds "$total_duration"
    echo "______Finish_________"
    echo "                   "
done

echo "                   "
echo "                   "

for ((i=0; i<${#size[@]}; i++)); do
    echo "FINAL results from resolution ${size[$i]} x ${size[$i]}:"
    convert_milliseconds "${times[$i]}"
done

echo "                   "
echo "                   "

global_end=$(date +%s%3N)
global_duration=$((global_end - global_start))
echo "TOTAL TIME OF THE EXPERMIENT: "
convert_milliseconds "$global_duration"

echo "                   "
echo "                   "
echo "Compute Finished..."
################################################################
##################################
###################
#######
###
#