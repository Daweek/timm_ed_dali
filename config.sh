#!/bin/bash
# Configuration file to define variables

# Modules to run on ABCI
# ======== Modules ========
echo "Include main ABCI modules for 3.0 .."
source /etc/profile.d/modules.sh
module purge
####### MPI
# module load intel-mpi/2021.13
# module load hpcx/2.20
module load hpcx-mt/2.20
# Load CUSTOM OpenMPI with CUDA support
# export PATH=$HOME/apps/openmpi/bin:$PATH
module load cuda/12.4/12.4.1 cudnn/9.5/9.5.1 nccl

# ======== Pyenv/ ========
echo "Include main python environment..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

pyenv local 3.11.11

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# Extra routines 
echo "Including extra util routines to measure time..."
convert_milliseconds() {
    local total_ms=$1

    # Calculate total seconds and remaining milliseconds
    local total_seconds=$(echo "$total_ms / 1000" | bc)
    local remaining_ms=$((total_ms % 1000))  # Remaining milliseconds

    # Calculate days, hours, minutes, and seconds using bc
    local days=$(echo "$total_seconds / 86400" | bc)
    local hours=$(echo "($total_seconds % 86400) / 3600" | bc)
    local minutes=$(echo "($total_seconds % 3600) / 60" | bc)
    local seconds=$(echo "$total_seconds % 60" | bc)

    # Print in D:H:M:S.ms format
    printf "\t%d days, %02d hours, %02d minutes, %02d seconds, %03d milliseconds\n" "$days" "$hours" "$minutes" "$seconds" "$remaining_ms"
}