#!/bin/bash
# Configuration file to define variables

# Modules to run on ABCI
# ======== Modules ========
echo "Include main ABCI modules.."
source /etc/profile.d/modules.sh
module purge
module load cuda/12.4/12.4.0 cudnn/9.1/9.1.1 nccl/2.21/2.21.5-1 gcc/13.2.0 cmake/3.29.0 hpcx-mt/2.12

# ======== Pyenv/ ========
echo "Include main python environment..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

pyenv local anaconda3-2023.07-2/envs/ffcv

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